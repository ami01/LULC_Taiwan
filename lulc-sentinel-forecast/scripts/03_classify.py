"""Step 3 (city sites): six classifiers x eight feature sets, reference year.

Design
    * Feature sets isolate what each Sentinel input adds: the 10 m bands alone,
      all ten bands, spectral indices, seasonality, and Sentinel-1 SAR, with
      Landsat 8/9 as the 30 m baseline the Sentinel missions are measured
      against.
    * Every model is tuned per feature set with the same budget, by
      RandomizedSearchCV over spatial-block-grouped folds of the training set.
    * Hold-out blocks are never seen in tuning. Intervals resample blocks;
      pairwise tests are clustered McNemar with Holm correction.
    * Temporal transfer: models trained on reference-year composites are scored
      on the same (stable) test pixels in the other epochs, which is what a
      change map built from one trained model relies on.

Outputs (results/<site>/tables/): classifier_featureset.csv, perclass_best.csv,
mcnemar_models.csv, mcnemar_featuresets.csv, temporal_transfer.csv,
best_config.json; data/<site>/features_<year>.csv; data/<site>/test_predictions.csv
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lulckit import stats as S  # noqa: E402
from lulckit.features import at_points, feature_sets  # noqa: E402
from lulckit.models import MODELS, Fitted  # noqa: E402
from lulckit.sites import CLASSES, site  # noqa: E402


def load_features(s, df, year):
    p = s["data"] / f"features_{year}.csv"
    if p.exists():
        return pd.read_csv(p)
    sensors = ("s2", "s1", "l8") if year == s["ref_year"] else ("s2", "s1")
    f = pd.DataFrame(at_points(s, year, df, sensors))
    f.to_csv(p, index=False, float_format="%.5g")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    s = site(a.site)
    df = pd.read_csv(s["data"] / "sample_points.csv")
    ry = s["ref_year"]
    F = {y: load_features(s, df, y) for y in s["years"]}
    tr, te = (df["split"] == "train").to_numpy(), (df["split"] == "test").to_numpy()
    y, blocks = df["y"].to_numpy(), df["block"].to_numpy()
    classes = sorted(np.unique(y))

    fsets = {k: v for k, v in feature_sets(s).items() if all(c in F[ry] for c in v)}
    missing = set(feature_sets(s)) - set(fsets)
    if missing:
        print(f"skipping feature sets with missing inputs: {sorted(missing)}")

    rows, preds, fitted = [], {}, {}
    for fs, cols in fsets.items():
        X = F[ry][cols].to_numpy("float32")
        for m in a.models:
            t = time.time()
            fm = Fitted(m, X[tr], y[tr], blocks[tr], n_iter=a.n_iter)
            p = fm.predict(X[te])
            preds[(fs, m)] = p
            fitted[(fs, m)] = fm
            r = dict(feature_set=fs, model=m, n_features=len(cols), cv_OA=fm.cv_score,
                     **S.scores(y[te], p), **S.block_bootstrap(y[te], p, blocks[te]),
                     **S.design_effect(p == y[te], blocks[te]),
                     fit_s=round(time.time() - t, 1), params=json.dumps(fm.best_params))
            rows.append(r)
            print(f"{fs:20s} {m:9s} OA={r['OA']:.4f} [{r['OA_lo']:.4f},{r['OA_hi']:.4f}] "
                  f"F1={r['macroF1']:.4f} deff={r['deff']:.1f} ({r['fit_s']} s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(s["tables"] / "classifier_featureset.csv", index=False, float_format="%.4f")

    pr = df.loc[te, ["row", "col", "y", "block"]].copy()
    for (fs, m), p in preds.items():
        pr[f"{fs}|{m}"] = p
    pr.to_csv(s["data"] / "test_predictions.csv", index=False)

    # Pairwise tests. Family 1: models within each feature set.
    yt, bt = y[te], blocks[te]
    out = []
    for fs in fsets:
        for ma, mb in itertools.combinations(a.models, 2):
            pa, pb = preds[(fs, ma)], preds[(fs, mb)]
            st, pc = S.mcnemar_clustered(yt, pa, pb, bt)
            _, pn = S.mcnemar_naive(yt, pa, pb)
            out.append(dict(feature_set=fs, A=ma, B=mb, OA_A=np.mean(pa == yt),
                            OA_B=np.mean(pb == yt), p_naive=pn, p_clustered=pc))
    mm = pd.DataFrame(out)
    mm["p_holm_naive"] = mm.groupby("feature_set")["p_naive"].transform(S.holm)
    mm["p_holm_clustered"] = mm.groupby("feature_set")["p_clustered"].transform(S.holm)
    mm.to_csv(s["tables"] / "mcnemar_models.csv", index=False, float_format="%.4g")

    # Family 2: feature sets within each model.
    out = []
    for m in a.models:
        for fa, fb in itertools.combinations(fsets, 2):
            pa, pb = preds[(fa, m)], preds[(fb, m)]
            st, pc = S.mcnemar_clustered(yt, pa, pb, bt)
            _, pn = S.mcnemar_naive(yt, pa, pb)
            out.append(dict(model=m, A=fa, B=fb, OA_A=np.mean(pa == yt),
                            OA_B=np.mean(pb == yt), p_naive=pn, p_clustered=pc))
    mf = pd.DataFrame(out)
    mf["p_holm_naive"] = mf.groupby("model")["p_naive"].transform(S.holm)
    mf["p_holm_clustered"] = mf.groupby("model")["p_clustered"].transform(S.holm)
    mf.to_csv(s["tables"] / "mcnemar_featuresets.csv", index=False, float_format="%.4g")

    # Operational configuration for mapping: the best 10 m Sentinel feature set
    # and model by hold-out OA (Landsat sets cannot be mapped at 10 m).
    sent = res[~res["feature_set"].str.startswith("L8")]
    best = sent.sort_values(["OA", "macroF1"], ascending=False).iloc[0]
    bfs, bm = best["feature_set"], best["model"]
    pc_tab = S.per_class(yt, preds[(bfs, bm)], classes)
    pc_tab.insert(1, "name", [CLASSES[k] for k in pc_tab["class_id"]])
    pc_tab.to_csv(s["tables"] / "perclass_best.csv", index=False, float_format="%.4f")
    cm = pd.DataFrame(confusion_matrix(yt, preds[(bfs, bm)], labels=classes),
                      index=[CLASSES[k] for k in classes], columns=[CLASSES[k] for k in classes])
    cm.to_csv(s["tables"] / "confusion_best.csv")

    # Temporal transfer of every Sentinel feature set with the best model.
    tt = []
    for fs, cols in fsets.items():
        if fs.startswith("L8"):
            continue
        fm = fitted[(fs, bm)]
        for yr in s["years"]:
            Xy = F[yr][cols].to_numpy("float32")
            p = fm.predict(Xy[te])
            tt.append(dict(feature_set=fs, model=bm, train_year=ry, test_year=yr,
                           **S.scores(yt, p)))
            if yr != ry:   # retrained on that year's own composites, same params
                fy = Fitted(bm, Xy[tr], y[tr], blocks[tr], n_iter=a.n_iter)
                py = fy.predict(Xy[te])
                tt.append(dict(feature_set=fs, model=bm, train_year=yr, test_year=yr,
                               **S.scores(yt, py)))
    pd.DataFrame(tt).to_csv(s["tables"] / "temporal_transfer.csv", index=False,
                            float_format="%.4f")

    json.dump(dict(feature_set=bfs, model=bm, OA=float(best["OA"]),
                   params=json.loads(best["params"]), classes=[int(c) for c in classes],
                   features=fsets[bfs]),
              open(s["tables"] / "best_config.json", "w"), indent=2)
    print(f"\nBest Sentinel configuration: {bfs} + {bm}, OA {best['OA']:.4f}")


if __name__ == "__main__":
    main()
