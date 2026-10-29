"""Step 4b (city sites): temporally consistent maps by change-gated updating.

Classifying each epoch independently and differencing the maps confounds
change with classification error: in Ljubljana the raw 2021 and 2024 maps
disagree on 11 % of 30 m cells, an order of magnitude more than plausible
three-year land change, and built-up area appears to shrink.

Here the reference-year map is kept and updated only where the imagery itself
says something changed:

1. For every pixel, the difference between the epoch's and the reference
   year's multi-season Sentinel-2 features is standardised by the robust
   spread (1.4826 x MAD) of the same difference at the stable sample pixels,
   and summarised as a root-mean-square change magnitude.
2. The threshold is the 99th percentile of that magnitude at the stable
   sample pixels, i.e. a 1 % false-alarm rate on land that did not change.
3. A pixel takes the epoch's own classified label if it exceeds the threshold,
   otherwise it keeps the reference-year label.

This is post-classification comparison gated by change-vector magnitude, the
logic of operational map updating (e.g. NLCD). Raw and gated change rates are
both reported.

Writes data/cache/<site>/mapc10_<year>.tif, overwrites map30_<year>.tif with the
gated version, and results/<site>/tables/change_gating.csv.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lulckit.features import S2_ALL, at_points, wall_to_wall  # noqa: E402
from lulckit.grid import coarsen, geobox, mode_downsample, read_tif, write_tif  # noqa: E402
from lulckit.sites import site  # noqa: E402

ROWS = 256
FALSE_ALARM = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    a = ap.parse_args()
    s = site(a.site)
    cfg = json.load(open(s["tables"] / "best_config.json"))
    ry = s["ref_year"]
    feats = [f"s2_{sea}_{b}" for sea in s["seasons"] for b in S2_ALL + ["NDVI", "NDBI", "MNDWI"]]
    df = pd.read_csv(s["data"] / "sample_points.csv")
    gb = geobox(s, 10, pad=200)
    f = s["forecast_res"] // 10
    ref, _, _ = read_tif(s["cache"] / f"map10_{ry}_{cfg['model']}.tif")
    Fr = pd.DataFrame(at_points(s, ry, df, ("s2",)))[feats].to_numpy("float32")
    rows = []
    write_tif(s["cache"] / f"mapc10_{ry}.tif", ref, gb, nodata=0)
    write_tif(s["cache"] / f"map30_{ry}.tif", mode_downsample(ref, f, nodata=0),
              coarsen(gb, f), nodata=0)
    for y in s["years"]:
        if y == ry:
            continue
        Fy = pd.DataFrame(at_points(s, y, df, ("s2",)))[feats].to_numpy("float32")
        d = Fy - Fr
        med = np.nanmedian(d, 0)
        scale = 1.4826 * np.nanmedian(np.abs(d - med), 0) + 1e-6
        mag_pts = np.sqrt(np.nanmean(((d - med) / scale) ** 2, 1))
        thr = float(np.nanpercentile(mag_pts, 100 * (1 - FALSE_ALARM)))
        raw, _, _ = read_tif(s["cache"] / f"map10_{y}_{cfg['model']}.tif")
        out = ref.copy()
        changed = np.zeros(ref.shape, bool)
        for r0 in range(0, gb.shape[0], ROWS):
            sl = slice(r0, min(r0 + ROWS, gb.shape[0]))
            cy = wall_to_wall(s, y, feats, sl)
            cr = wall_to_wall(s, ry, feats, sl)
            dd = (cy - cr - med[:, None, None]) / scale[:, None, None]
            mag = np.sqrt(np.nanmean(dd ** 2, 0))
            changed[sl] = np.nan_to_num(mag, nan=0) > thr
        valid = (ref > 0) & (raw > 0)
        upd = valid & changed
        out[upd] = raw[upd]
        write_tif(s["cache"] / f"mapc10_{y}.tif", out, gb, nodata=0)
        write_tif(s["cache"] / f"map30_{y}.tif", mode_downsample(out, f, nodata=0),
                  coarsen(gb, f), nodata=0)
        rows.append(dict(year=y, ref_year=ry, threshold=thr,
                         frac_flagged=float(changed[valid].mean()),
                         raw_label_change=float((raw[valid] != ref[valid]).mean()),
                         gated_label_change=float((out[valid] != ref[valid]).mean()),
                         raw_built_km2=float(np.sum(raw == 2) * 1e-4),
                         gated_built_km2=float(np.sum(out == 2) * 1e-4),
                         ref_built_km2=float(np.sum(ref == 2) * 1e-4)))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()},
              flush=True)
    pd.DataFrame(rows).to_csv(s["tables"] / "change_gating.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
