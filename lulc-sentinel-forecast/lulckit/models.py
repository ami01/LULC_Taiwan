"""The six classifiers and their tuning spaces (same family as the Taipei study)."""

import warnings

import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42

MODELS = {
    "CART": (DecisionTreeClassifier(random_state=SEED),
             {"max_depth": randint(4, 30), "min_samples_leaf": randint(1, 20),
              "criterion": ["gini", "entropy"]}, False),
    "RF": (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=2),
           {"max_features": ["sqrt", "log2", 0.3], "min_samples_leaf": randint(1, 6),
            "max_depth": [None, 20, 30]}, False),
    "XGBoost": (XGBClassifier(n_estimators=300, tree_method="hist", random_state=SEED,
                              n_jobs=2, verbosity=0),
                {"max_depth": randint(3, 10), "learning_rate": loguniform(0.03, 0.3),
                 "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5)}, False),
    "LightGBM": (LGBMClassifier(n_estimators=300, random_state=SEED, n_jobs=2, verbose=-1),
                 {"num_leaves": randint(15, 127), "learning_rate": loguniform(0.03, 0.3),
                  "subsample": uniform(0.6, 0.4), "subsample_freq": [1],
                  "colsample_bytree": uniform(0.5, 0.5), "min_child_samples": randint(5, 40)},
                 False),
    "SVM-RBF": (SVC(kernel="rbf", random_state=SEED),
                {"C": loguniform(0.5, 500), "gamma": loguniform(1e-3, 1)}, True),
    "MLP": (MLPClassifier(max_iter=600, early_stopping=True, random_state=SEED),
            {"hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 64)],
             "alpha": loguniform(1e-5, 1e-2), "learning_rate_init": loguniform(1e-4, 1e-2)},
            True),
}


def make(name):
    est, space, scale = MODELS[name]
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    steps.append(("m", est))
    return Pipeline(steps), {f"m__{k}": v for k, v in space.items()}


class Fitted:
    """A tuned model plus the label encoder XGBoost needs (labels 0..k-1)."""

    def __init__(self, name, X, y, groups, n_iter=12, folds=3):
        self.le = LabelEncoder().fit(y)
        pipe, space = make(name)
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
        self.search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=cv,
                                         scoring="accuracy", random_state=SEED,
                                         n_jobs=2, refit=True)
        self.search.fit(X, self.le.transform(y), groups=groups)
        self.best_params = {k.replace("m__", ""): (v.item() if hasattr(v, "item") else v)
                            for k, v in self.search.best_params_.items()}
        self.cv_score = float(self.search.best_score_)

    def predict(self, X):
        return self.le.inverse_transform(self.search.predict(X))

    def predict_chunked(self, X, chunk=500_000):
        out = np.empty(len(X), dtype=self.le.classes_.dtype)
        for i in range(0, len(X), chunk):
            out[i:i + chunk] = self.predict(X[i:i + chunk])
        return out
