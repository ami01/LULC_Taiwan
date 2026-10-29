"""Land-change forecasting: Markov quantity, ML transition potential, CA allocation.

The model separates the three questions every land-change model answers:

1. How much?   Markov chain. The observed transition matrix over an interval
               is converted to an annual matrix (principal matrix root,
               negative entries clipped and rows renormalised) and projected.
2. Where?      A transition-potential surface per target class: the probability
               that a cell not in class k at t0 is class k at t1, learned from
               the drivers by logistic regression (CLUE-S / CA-Markov style),
               a multilayer perceptron (ANN-CA / Land Change Modeler style) or a
               random forest (PLUS style).
3. Allocation  A demand-constrained cellular automaton in the CLUE-S / Dyna-CLUE
               tradition. Each year, every cell gets a score per class:
                   stay in current class c:   elas_c + iter_c
                   convert to class k != c:   G_k(p) * N_k(p) + iter_k
               where G_k is the transition potential, N_k = (1 - w) + w * Omega_k
               mixes in the share of class k in the 5 x 5 neighbourhood, elas_c
               is the class's resistance to conversion and iter_k is an
               iteration variable adjusted until allocated areas meet the
               Markov demand. Each cell takes its highest-scoring class.

Validation follows Pontius et al.: three maps (initial, observed, simulated)
give hits, misses, wrong hits and false alarms, and the figure of merit
FoM = H / (M + H + W + F), always reported next to a persistence null model and
a Markov-quantity / random-allocation model.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.linalg import fractional_matrix_power
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


# --------------------------------------------------------------------------
# 1. Markov quantity
# --------------------------------------------------------------------------
def transition_counts(m0, m1, classes, mask):
    K = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    lut = np.full(256, -1)
    for c, i in idx.items():
        lut[c] = i
    a, b = lut[m0[mask]], lut[m1[mask]]
    ok = (a >= 0) & (b >= 0)
    C = np.zeros((K, K), dtype=np.int64)
    np.add.at(C, (a[ok], b[ok]), 1)
    return C


def transition_matrix(C):
    rs = C.sum(1, keepdims=True)
    P = np.where(rs > 0, C / np.maximum(rs, 1), 0.0)
    for i in np.flatnonzero(rs[:, 0] == 0):
        P[i, i] = 1.0
    return P


def annualize(P, years):
    """Annual transition matrix from a ``years``-long one (principal root).

    The root of a stochastic matrix need not be stochastic (the embedding
    problem); negative entries are clipped and rows renormalised, the simplest
    of the regularisations compared by Kreinin & Sidelnikova (2001).
    """
    A = np.real(fractional_matrix_power(P, 1.0 / years))
    A = np.clip(A, 0, None)
    return A / A.sum(1, keepdims=True)


def project(areas0, P_annual, years):
    return areas0 @ np.linalg.matrix_power(P_annual, int(years))


def class_areas(m, classes, mask):
    return np.array([np.sum(m[mask] == c) for c in classes], dtype=float)


# --------------------------------------------------------------------------
# 2. Transition potential
# --------------------------------------------------------------------------
def dist_to_class(m, c, res):
    """Euclidean distance (m) to the nearest cell of class c."""
    if not np.any(m == c):
        return np.full(m.shape, 1e6, dtype="float32")
    return (ndimage.distance_transform_edt(m != c) * res).astype("float32")


def _make(kind):
    if kind == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    if kind == "MLP":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                           max_iter=500, early_stopping=True,
                                           random_state=SEED))
    if kind == "RF":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                      max_features="sqrt", n_jobs=4, random_state=SEED)
    raise ValueError(kind)


class Potential:
    """One binary 'becomes class k' model per target class."""

    def __init__(self, kind, classes, res, driver_names, dynamic=True,
                 n_per=10000, seed=SEED):
        self.kind, self.classes, self.res = kind, list(classes), res
        self.driver_names, self.dynamic = list(driver_names), dynamic
        self.n_per, self.rng = n_per, np.random.default_rng(seed)
        self.models, self.auc_in, self.importance = {}, {}, {}

    def _X(self, drivers, m0, k, idx):
        cols = [drivers[n].ravel()[idx] for n in self.driver_names]
        if self.dynamic:
            cols.append(dist_to_class(m0, k, self.res).ravel()[idx])
        return np.column_stack(cols).astype("float32")

    def feature_names(self):
        return self.driver_names + (["dist_to_class"] if self.dynamic else [])

    def fit(self, m0, m1, drivers, mask, allowed):
        flat0, flat1, fm = m0.ravel(), m1.ravel(), mask.ravel()
        for k in self.classes:
            ok_from = np.isin(flat0, [c for c in self.classes if c != k and allowed[(c, k)]])
            cand = fm & ok_from
            pos = np.flatnonzero(cand & (flat1 == k))
            neg = np.flatnonzero(cand & (flat1 != k))
            if len(pos) < 30 or len(neg) < 30:
                self.models[k] = None
                continue
            n = min(self.n_per, len(pos))
            p = self.rng.choice(pos, n, replace=False)
            q = self.rng.choice(neg, min(self.n_per, len(neg)), replace=False)
            idx = np.concatenate([p, q])
            y = np.r_[np.ones(len(p)), np.zeros(len(q))]
            X = self._X(drivers, m0, k, idx)
            # 25 % internal hold-out for AUC and permutation importance.
            perm = self.rng.permutation(len(y))
            cut = int(0.75 * len(y))
            tr, te = perm[:cut], perm[cut:]
            mdl = _make(self.kind).fit(X[tr], y[tr])
            self.auc_in[k] = float(roc_auc_score(y[te], mdl.predict_proba(X[te])[:, 1]))
            if self.kind == "RF":
                pi = permutation_importance(mdl, X[te], y[te], n_repeats=5,
                                            random_state=SEED, scoring="roc_auc", n_jobs=1)
                self.importance[k] = dict(zip(self.feature_names(), pi.importances_mean))
            self.models[k] = _make(self.kind).fit(X, y)
        return self

    def surfaces(self, m0, drivers, mask):
        """Potential G_k for every cell (0 where k cannot be reached)."""
        G = np.zeros((len(self.classes),) + m0.shape, dtype="float32")
        idx = np.flatnonzero(mask.ravel())
        for i, k in enumerate(self.classes):
            if self.models.get(k) is None:
                continue
            X = self._X(drivers, m0, k, idx)
            g = np.empty(len(idx), dtype="float32")
            for s in range(0, len(idx), 400_000):
                g[s:s + 400_000] = self.models[k].predict_proba(X[s:s + 400_000])[:, 1]
            G[i].ravel()[idx] = g
        return G

    def auc_out(self, m0, m1, drivers, mask, allowed, n=20000):
        """Out-of-time AUC: potentials from this model, changes of another period."""
        out = {}
        flat0, flat1, fm = m0.ravel(), m1.ravel(), mask.ravel()
        for k in self.classes:
            if self.models.get(k) is None:
                continue
            ok_from = np.isin(flat0, [c for c in self.classes if c != k and allowed[(c, k)]])
            cand = fm & ok_from
            pos = np.flatnonzero(cand & (flat1 == k))
            neg = np.flatnonzero(cand & (flat1 != k))
            if len(pos) < 30:
                continue
            p = self.rng.choice(pos, min(n, len(pos)), replace=False)
            q = self.rng.choice(neg, min(n, len(neg)), replace=False)
            idx = np.concatenate([p, q])
            y = np.r_[np.ones(len(p)), np.zeros(len(q))]
            out[k] = float(roc_auc_score(y, self.models[k].predict_proba(
                self._X(drivers, m0, k, idx))[:, 1]))
        return out


# --------------------------------------------------------------------------
# 3. Allocation
# --------------------------------------------------------------------------
def neighbourhood(m, classes, size=5):
    out = np.empty((len(classes),) + m.shape, dtype="float32")
    for i, c in enumerate(classes):
        out[i] = ndimage.uniform_filter((m == c).astype("float32"), size=size, mode="nearest")
    return out


def allocate(m0, G, demand, classes, mask, elas, allowed, w=0.5, restrict=None,
             forbid=None, max_iter=400, tol=0.002, it0=None):
    """One time step of demand-constrained competitive allocation.

    ``demand`` is the target cell count per class; ``restrict`` marks cells that
    may not change in this scenario; ``forbid`` maps a class to cells that may
    not convert *into* it. Returns (new map, iteration variables).
    """
    K = len(classes)
    cur = np.full(m0.shape, -1, dtype=np.int16)
    for i, c in enumerate(classes):
        cur[m0 == c] = i
    valid = mask & (cur >= 0)
    frozen = ~valid if restrict is None else (~valid | restrict)
    Nb = neighbourhood(m0, classes)
    base = G * ((1 - w) + w * Nb)            # conversion score without iter
    el = np.array([elas[c] for c in classes], dtype="float32")
    # Disallowed transitions get -inf.
    allow = np.array([[allowed[(a, b)] for b in classes] for a in classes])
    curv = cur[valid]
    S0 = base[:, valid].T.copy()             # n x K
    S0[~allow[curv]] = -np.inf
    S0[np.arange(len(curv)), curv] = el[curv]
    for c, fmask in (forbid or {}).items():
        i = classes.index(c)
        col = S0[:, i]
        col[fmask[valid] & (curv != i)] = -np.inf
    itv = np.zeros(K, dtype="float32") if it0 is None else it0.astype("float32").copy()
    # Frozen-but-valid cells keep their class and count toward the totals.
    fz = frozen[valid]
    target = np.asarray(demand, float)
    step = 0.05
    for it in range(max_iter):
        choice = np.argmax(S0 + itv, axis=1)
        choice[fz] = curv[fz]
        alloc = np.bincount(choice, minlength=K)
        err = (target - alloc) / np.maximum(target, 1)
        if np.max(np.abs(err)) < tol:
            break
        itv += step * np.clip(err, -1, 1)
        if it % 100 == 99:
            step *= 0.5
    out = m0.copy()
    out[valid] = np.asarray(classes, dtype=m0.dtype)[choice]
    return out, itv


def simulate(m_start, P_annual, years, G, classes, mask, elas, allowed, w=0.5,
             restrict=None, forbid=None, demand_override=None, return_steps=(), step=1):
    """Simulation in ``step``-year increments; potentials G held fixed.

    Regional maps are 5-10 years apart, so stepping a year at a time adds cost
    without information; the neighbourhood is refreshed at each step. A final
    partial step is taken if ``years`` is not a multiple of ``step``.
    """
    m = m_start.copy()
    a = class_areas(m, classes, mask)
    itv, keep = None, {}
    times = list(range(step, int(years) + 1, step))
    times = sorted(set(times) | {int(years)} | {t for t in return_steps if t <= years})
    prev = 0
    for t in times:
        a = a @ np.linalg.matrix_power(P_annual, t - prev)
        prev = t
        d = a if demand_override is None else demand_override(t, a)
        fb = forbid(m) if callable(forbid) else forbid
        m, itv = allocate(m, G, d, classes, mask, elas, allowed, w=w,
                          restrict=restrict, forbid=fb, it0=itv)
        if t in return_steps:
            keep[t] = m.copy()
    return m, keep


def random_allocation(m_start, demand, classes, mask, allowed, seed=SEED):
    """Markov quantity, random location: the 'quantity-only' baseline."""
    rng = np.random.default_rng(seed)
    m = m_start.copy()
    cur = class_areas(m, classes, mask)
    gain = np.maximum(np.asarray(demand) - cur, 0)
    loss = np.maximum(cur - np.asarray(demand), 0)
    donors = []
    for i, c in enumerate(classes):
        if loss[i] > 0:
            cells = np.flatnonzero((m.ravel() == c) & mask.ravel())
            donors.append(rng.choice(cells, int(round(loss[i])), replace=False))
    pool = rng.permutation(np.concatenate(donors)) if donors else np.array([], int)
    flat, pos = m.ravel(), 0
    for i, c in enumerate(classes):
        n = int(round(gain[i]))
        take = pool[pos:pos + n]
        take = take[[allowed[(flat[j], c)] for j in take]] if n else take
        flat[take] = c
        pos += n
    return flat.reshape(m.shape)


# --------------------------------------------------------------------------
# 4. Validation
# --------------------------------------------------------------------------
def pontius(m0, obs, sim, mask, classes=None, focus=None):
    """Three-map comparison. ``focus`` restricts to transitions *into* a class."""
    a, b, c = m0[mask], obs[mask], sim[mask]
    if focus is not None:
        oc = (b != a) & (b == focus)
        sc = (c != a) & (c == focus)
        H = np.sum(oc & sc)
        M = np.sum(oc & ~sc)
        F = np.sum(~oc & sc)
        W = 0
    else:
        oc, sc = b != a, c != a
        H = np.sum(oc & sc & (b == c))
        W = np.sum(oc & sc & (b != c))
        M = np.sum(oc & ~sc)
        F = np.sum(~oc & sc)
    n = a.size
    fom = H / max(1, H + M + W + F)
    PA = H / max(1, H + M + W)
    UA = H / max(1, H + W + F)
    out = dict(hits=int(H), misses=int(M), wrong_hits=int(W), false_alarms=int(F),
               FoM=float(fom), producers=float(PA), users=float(UA),
               obs_change=float(np.mean(oc)), sim_change=float(np.mean(sc)))
    if focus is None and classes is not None:
        qd = sum(abs(np.sum(b == k) - np.sum(c == k)) for k in classes) / (2 * n)
        agree = float(np.mean(b == c))
        out.update(agreement=agree, quantity_dis=float(qd),
                   allocation_dis=float(1 - agree - qd),
                   null_agreement=float(np.mean(a == b)))
    return out


def multires_agreement(obs, sim, mask, classes, factors=(1, 3, 9, 27)):
    """Fuzzy agreement at coarser resolutions (Pontius, Huffaker & Denman 2004)."""
    rows = []
    for f in factors:
        h, w = (obs.shape[0] // f) * f, (obs.shape[1] // f) * f
        mk = mask[:h, :w].reshape(h // f, f, w // f, f).sum((1, 3))
        tot = 0.0
        for k in classes:
            ok = ((obs[:h, :w] == k) & mask[:h, :w]).reshape(h // f, f, w // f, f).sum((1, 3))
            sk = ((sim[:h, :w] == k) & mask[:h, :w]).reshape(h // f, f, w // f, f).sum((1, 3))
            tot += np.minimum(ok, sk).sum()
        rows.append(dict(factor=f, agreement=float(tot / max(1, mk.sum()))))
    return pd.DataFrame(rows)
