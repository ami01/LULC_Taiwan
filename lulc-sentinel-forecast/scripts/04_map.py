"""Step 4 (city sites): wall-to-wall maps for every epoch.

Each epoch is classified by the best Sentinel configuration from step 3,
retrained on that epoch's own composites and the training split of the stable
sample, with the hyper-parameters tuned in the reference year. Also writes the
30 m majority-aggregated maps used by the forecast, and per-epoch class areas.
For the reference year, the remaining tree models are mapped too, so map-level
disagreement between classifiers can be shown (SVM and MLP are skipped: their
per-pixel cost on 10^7 pixels is out of proportion to what they would add).

Usage:
    python scripts/04_map.py --site varanasi
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lulckit.features import wall_to_wall  # noqa: E402
from lulckit.grid import aoi_mask, coarsen, geobox, mode_downsample, write_tif  # noqa: E402
from lulckit.models import make  # noqa: E402
from lulckit.sites import CLASSES, site  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

ROWS = 256


def fit_fixed(model, params, X, y):
    pipe, _ = make(model)
    pipe.set_params(**{f"m__{k}": v for k, v in params.items()})
    le = LabelEncoder().fit(y)
    pipe.fit(X, le.transform(y))
    return pipe, le


def classify(s, year, cols, pipe, le, gb, mask):
    out = np.zeros(gb.shape, dtype="uint8")
    for r0 in range(0, gb.shape[0], ROWS):
        sl = slice(r0, min(r0 + ROWS, gb.shape[0]))
        cube = wall_to_wall(s, year, cols, sl)
        n, h, w = cube.shape
        X = cube.reshape(n, -1).T
        # A pixel cloud-masked in one season still has the others; the
        # pipeline's median imputer fills the gap, as it did in training.
        m = mask[sl].ravel() & np.isfinite(X).any(1)
        pred = np.zeros(h * w, dtype="uint8")
        if m.any():
            pred[m] = le.inverse_transform(pipe.predict(X[m])).astype("uint8")
        out[sl] = pred.reshape(h, w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    a = ap.parse_args()
    s = site(a.site)
    cfg = json.load(open(s["tables"] / "best_config.json"))
    res = pd.read_csv(s["tables"] / "classifier_featureset.csv")
    df = pd.read_csv(s["data"] / "sample_points.csv")
    tr = (df["split"] == "train").to_numpy()
    gb = geobox(s, 10, pad=200)
    mask = aoi_mask(s, gb)
    f = s["forecast_res"] // 10
    gb30 = coarsen(gb, f)
    cols = cfg["features"]

    jobs = [(y, cfg["model"]) for y in s["years"]]
    others = ["RF", "XGBoost", "LightGBM", "CART"]
    jobs += [(s["ref_year"], m) for m in others if m != cfg["model"]]
    rows = []
    for y, model in jobs:
        out = s["cache"] / f"map10_{y}_{model}.tif"
        F = pd.read_csv(s["data"] / f"features_{y}.csv")
        params = json.loads(res[(res.feature_set == cfg["feature_set"]) &
                                (res.model == model)]["params"].iloc[0])
        pipe, le = fit_fixed(model, params, F.loc[tr, cols].to_numpy("float32"),
                             df.loc[tr, "y"].to_numpy())
        m = classify(s, y, cols, pipe, le, gb, mask)
        write_tif(out, m, gb, nodata=0)
        for k in cfg["classes"]:
            rows.append(dict(year=y, model=model, class_id=k, name=CLASSES[k],
                             area_km2=float(np.sum(m == k) * 1e-4),
                             share=float(np.sum(m == k) / mask.sum())))
        if model == cfg["model"]:
            write_tif(s["cache"] / f"map30_{y}.tif", mode_downsample(m, f, nodata=0),
                      gb30, nodata=0)
        print(f"  {y} {model} mapped", flush=True)
    pd.DataFrame(rows).to_csv(s["tables"] / "map_areas.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
