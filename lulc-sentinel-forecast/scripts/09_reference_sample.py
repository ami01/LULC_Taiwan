"""Step 9 (city sites): probability reference sample for an unbiased accuracy check.

The hold-out accuracy in step 3 is measured on stable-consensus pixels and is an
upper bound. The only way to an unbiased estimate is a probability sample from
the map, labelled independently. This script draws one, following Olofsson et
al. (2014): stratified random by mapped class, total n from the target standard
error of overall accuracy, at least ``--min-per-class`` per stratum.

    python scripts/09_reference_sample.py --site varanasi draw
        -> data/<site>/reference_sample.csv / .geojson with an empty
           `ref_class` column. Label each point by visual interpretation of
           high-resolution imagery for the map year (Google Earth historical
           imagery works), without looking at the map class.

    python scripts/09_reference_sample.py --site varanasi estimate
        -> results/<site>/tables/olofsson_accuracy.csv and olofsson_area.csv:
           area-weighted OA, user's and producer's accuracy, and stratified
           area estimates with 95 % confidence intervals.
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lulckit.grid import read_tif  # noqa: E402
from lulckit.sites import CLASSES, site  # noqa: E402


def sample_size(W, U, se_target):
    """Cochran / Olofsson eq. 13: n = (sum W_i S_i / S(O))^2."""
    S = np.sqrt(U * (1 - U))
    return int(np.ceil((np.sum(W * S) / se_target) ** 2))


def draw(s, year, se, min_per, seed):
    cfg = json.load(open(s["tables"] / "best_config.json"))
    m, gb, _ = read_tif(s["cache"] / f"map10_{year}_{cfg['model']}.tif")
    classes = [k for k in cfg["classes"] if np.any(m == k)]
    N = np.array([np.sum(m == k) for k in classes], float)
    W = N / N.sum()
    pc = pd.read_csv(s["tables"] / "perclass_best.csv").set_index("class_id")
    # Conjectured user's accuracy: consensus-sample UA minus 0.10, as a hedge
    # against its known optimism.
    U = np.array([max(0.6, pc.loc[k, "UA"] - 0.10) for k in classes])
    n = sample_size(W, U, se)
    alloc = np.maximum(min_per, np.round(W * n)).astype(int)
    rng = np.random.default_rng(seed)
    rows = []
    for k, nk in zip(classes, alloc):
        r, c = np.nonzero(m == k)
        pick = rng.choice(len(r), nk, replace=False)
        x = gb.affine.c + (c[pick] + 0.5) * gb.affine.a
        y = gb.affine.f + (r[pick] + 0.5) * gb.affine.e
        rows.append(pd.DataFrame({"x": x, "y": y, "map_class": k}))
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "pid", np.arange(1, len(df) + 1))
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=gb.crs).to_crs(4326)
    g["lon"], g["lat"] = g.geometry.x.round(6), g.geometry.y.round(6)
    g["ref_class"] = ""
    g["interpreter"], g["imagery_date"], g["note"] = "", "", ""
    # The interpreter sheet hides the map class: blind interpretation.
    g[["pid", "lon", "lat", "ref_class", "interpreter", "imagery_date", "note"]].to_csv(
        s["data"] / "reference_sample.csv", index=False)
    g[["pid", "map_class", "geometry"]].to_file(s["data"] / "reference_sample_key.geojson")
    design = pd.DataFrame(dict(class_id=classes, name=[CLASSES[k] for k in classes],
                               map_pixels=N.astype(int), weight=W, conjectured_UA=U, n=alloc))
    design.to_csv(s["tables"] / "reference_sample_design.csv", index=False, float_format="%.4f")
    print(design.to_string(index=False))
    print(f"total n = {alloc.sum()} (Cochran n = {n} for SE(OA) = {se})")


def estimate(s, year):
    cfg = json.load(open(s["tables"] / "best_config.json"))
    m = read_tif(s["cache"] / f"map10_{year}_{cfg['model']}.tif")[0]
    key = gpd.read_file(s["data"] / "reference_sample_key.geojson")
    sheet = pd.read_csv(s["data"] / "reference_sample.csv")
    d = key.merge(sheet, on="pid")
    d = d[d.ref_class.notna() & (d.ref_class.astype(str) != "")]
    d["ref_class"] = d.ref_class.astype(int)
    classes = sorted(set(cfg["classes"]))
    N = np.array([np.sum(m == k) for k in classes], float)
    W = N / N.sum()
    K = len(classes)
    nmat = np.zeros((K, K))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            nmat[i, j] = np.sum((d.map_class == a) & (d.ref_class == b))
    ni = nmat.sum(1)
    p = (W[:, None] * nmat / np.maximum(ni[:, None], 1))
    OA = np.trace(p)
    UA = np.diag(p) / np.maximum(p.sum(1), 1e-12)
    PA = np.diag(p) / np.maximum(p.sum(0), 1e-12)
    var_oa = np.sum(W ** 2 * UA * (1 - UA) / np.maximum(ni - 1, 1))
    area = p.sum(0) * N.sum() * 1e-4  # km2 at 10 m
    se_area = np.array([np.sqrt(np.sum(W ** 2 * (nmat[:, j] / np.maximum(ni, 1)) *
                                       (1 - nmat[:, j] / np.maximum(ni, 1)) /
                                       np.maximum(ni - 1, 1))) for j in range(K)]) * N.sum() * 1e-4
    pd.DataFrame(dict(class_id=classes, name=[CLASSES[k] for k in classes], UA=UA, PA=PA,
                      n_mapped=ni)).assign(OA=OA, OA_ci95=1.96 * np.sqrt(var_oa)).to_csv(
        s["tables"] / "olofsson_accuracy.csv", index=False, float_format="%.4f")
    pd.DataFrame(dict(class_id=classes, name=[CLASSES[k] for k in classes],
                      mapped_km2=N * 1e-4, estimated_km2=area, ci95_km2=1.96 * se_area)).to_csv(
        s["tables"] / "olofsson_area.csv", index=False, float_format="%.3f")
    print(f"OA = {OA:.4f} +/- {1.96 * np.sqrt(var_oa):.4f} (n = {len(d)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("action", choices=["draw", "estimate"])
    ap.add_argument("--year", type=int)
    ap.add_argument("--se", type=float, default=0.015)
    ap.add_argument("--min-per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    s = site(a.site)
    year = a.year or s["years"][-1]
    if a.action == "draw":
        draw(s, year, a.se, a.min_per_class, a.seed)
    else:
        estimate(s, year)


if __name__ == "__main__":
    main()
