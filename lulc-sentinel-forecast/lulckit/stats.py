"""Accuracy statistics that respect the spatial clustering of the sample.

Sample pixels are drawn at random, but pixels in the same spatial block share
neighbourhood, illumination and often the same field or building block, so
they are not independent. Intervals are computed by resampling whole blocks,
and pairwise tests use the clustered McNemar statistic of Durkalski et al.
(2003, Statistics in Medicine 22:2417-2428), as in the Taipei study
(taipei-lulc/src/mcnemar_test.py).
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score


def scores(y, p):
    return dict(OA=float(np.mean(y == p)),
                macroF1=float(f1_score(y, p, average="macro")),
                kappa=float(cohen_kappa_score(y, p)))


def block_bootstrap(y, p, blocks, n=1000, seed=42):
    """95 % percentile intervals for OA and macro-F1, resampling blocks."""
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    idx_by_block = {b: np.flatnonzero(blocks == b) for b in ub}
    oa, f1 = [], []
    for _ in range(n):
        pick = rng.choice(ub, len(ub), replace=True)
        ii = np.concatenate([idx_by_block[b] for b in pick])
        oa.append(np.mean(y[ii] == p[ii]))
        f1.append(f1_score(y[ii], p[ii], average="macro"))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return dict(OA_lo=q(oa)[0], OA_hi=q(oa)[1], F1_lo=q(f1)[0], F1_hi=q(f1)[1])


def design_effect(correct, blocks):
    """Kish design effect of per-pixel correctness under block clustering.

    ICC from the one-way ANOVA estimator; deff = 1 + (m0 - 1) * ICC with m0 the
    size-weighted mean cluster size.
    """
    df = pd.DataFrame({"c": correct.astype(float), "b": blocks})
    g = df.groupby("b")["c"]
    k, n = g.ngroups, len(df)
    m = g.size().to_numpy()
    m0 = (n - np.sum(m ** 2) / n) / (k - 1)
    grand = df["c"].mean()
    msb = np.sum(m * (g.mean().to_numpy() - grand) ** 2) / (k - 1)
    msw = np.sum((df["c"] - g.transform("mean")) ** 2) / (n - k)
    icc = (msb - msw) / (msb + (m0 - 1) * msw) if (msb + (m0 - 1) * msw) > 0 else 0.0
    icc = max(0.0, float(icc))
    deff = 1 + (m0 - 1) * icc
    return dict(icc=icc, deff=float(deff), n_eff=float(n / deff))


def mcnemar_clustered(y, pa, pb, clusters):
    a_ok, b_ok = pa == y, pb == y
    d = (~a_ok & b_ok).astype(int) - (a_ok & ~b_ok).astype(int)
    dk = pd.Series(d).groupby(np.asarray(clusters)).sum().to_numpy(float)
    denom = float(np.sum(dk ** 2))
    if denom == 0:
        return 0.0, 1.0
    stat = float(np.sum(dk) ** 2 / denom)
    return stat, float(stats.chi2.sf(stat, 1))


def mcnemar_naive(y, pa, pb):
    a_ok, b_ok = pa == y, pb == y
    n01, n10 = int(np.sum(~a_ok & b_ok)), int(np.sum(a_ok & ~b_ok))
    if n01 + n10 == 0:
        return 0.0, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(stat), float(stats.chi2.sf(stat, 1))


def holm(p):
    p = np.asarray(p, float)
    m, order = len(p), np.argsort(p)
    adj, run = np.empty(m), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def per_class(y, p, classes):
    rows = []
    for k in classes:
        tp = np.sum((y == k) & (p == k))
        ua = tp / max(1, np.sum(p == k))
        pa = tp / max(1, np.sum(y == k))
        rows.append(dict(class_id=k, UA=ua, PA=pa,
                         F1=2 * ua * pa / (ua + pa) if ua + pa else 0.0))
    return pd.DataFrame(rows)
