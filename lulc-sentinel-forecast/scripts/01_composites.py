"""Step 1 (city sites): seasonal Sentinel-2, Sentinel-1 and Landsat composites.

Writes to data/cache/<site>/:
    s2_<year>_<season>.tif     10 bands, reflectance x 10000, int16, 10 m
    s1_<year>_<season>.tif     VV, VH, VV-VH in dB x 100, int16, 10 m
    l8_<year>_<season>.tif     6 bands, reflectance x 10000, int16, 30 m (ref year only)
and a scene log to results/<site>/tables/composite_scenes.csv.

Usage:
    python scripts/01_composites.py --site varanasi
    python scripts/01_composites.py --site ljubljana --years 2021 --seasons summer
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lulckit import composite as cp  # noqa: E402
from lulckit.grid import geobox, write_tif  # noqa: E402
from lulckit.sites import site  # noqa: E402


def to_i16(a, scale):
    a = np.nan_to_num(np.asarray(a, dtype="float32") * scale, nan=-32768)
    return np.clip(a, -32768, 32767).astype("int16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--seasons", nargs="*")
    ap.add_argument("--sensors", nargs="*", default=["s2", "s1", "l8"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    s = site(a.site)
    gb10 = geobox(s, 10, pad=200)
    gb30 = geobox(s, 30, pad=210)
    print(f"{s['name']}: 10 m grid {gb10.shape}, 30 m grid {gb30.shape}")

    log_path = s["tables"] / "composite_scenes.csv"
    log = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    years = a.years or s["years"]
    seasons = a.seasons or list(s["seasons"])
    for y in years:
        for sea in seasons:
            m0, m1 = s["seasons"][sea]
            start, end = f"{y}-{m0}", f"{y}-{m1}"
            jobs = []
            if "s2" in a.sensors:
                jobs.append(("s2", gb10))
            if "s1" in a.sensors:
                jobs.append(("s1", gb10))
            if "l8" in a.sensors and y == s["ref_year"]:
                jobs.append(("l8", gb30))
            for sensor, gb in jobs:
                out = s["cache"] / f"{sensor}_{y}_{sea}.tif"
                if out.exists() and not a.force:
                    continue
                t = time.time()
                if sensor == "s2":
                    da, n, nobs = cp.s2_composite(gb, start, end)
                    write_tif(out, to_i16(da.values, 10000), gb, nodata=-32768,
                              names=list(da.band.values))
                    frac = float((np.asarray(nobs) > 0).mean())
                elif sensor == "s1":
                    da, n = cp.s1_composite(gb, start, end)
                    write_tif(out, to_i16(da.values, 100), gb, nodata=-32768,
                              names=list(da.band.values))
                    frac = float(np.isfinite(da.values[0]).mean())
                else:
                    da, n = cp.landsat_composite(gb, start, end)
                    write_tif(out, to_i16(da.values, 10000), gb, nodata=-32768,
                              names=list(da.band.values))
                    frac = float(np.isfinite(da.values[0]).mean())
                dt = time.time() - t
                print(f"  {sensor} {y} {sea}: {n} dates, {frac:.3f} covered, {dt:.0f} s")
                log = [r for r in log if not (r["sensor"] == sensor and r["year"] == y
                                              and r["season"] == sea)]
                log.append(dict(sensor=sensor, year=y, season=sea, start=start,
                                end=end, n_dates=n, frac_covered=round(frac, 4)))
                pd.DataFrame(log).sort_values(["sensor", "year", "season"]).to_csv(
                    log_path, index=False)


if __name__ == "__main__":
    main()
