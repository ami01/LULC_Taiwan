"""Step 8 (Slovenia, Ljubljana): independent check against LUCAS in-situ data.

LUCAS (Land Use/Cover Area frame Survey) points are visited on the ground by
Eurostat surveyors. The Copernicus modules (2018: d'Andrimont et al. 2021;
2022) record land cover around each point in a form intended for validating
Earth-observation products. They are the only reference data in this study
that were not themselves derived from satellite maps.

Caveats written into the papers: the Copernicus subset favours homogeneous
surroundings, so it under-represents fragmented built-up land; and it is a
subsample of a stratified design, not a simple random sample, so the scores
below describe agreement at these points, not unbiased map accuracy.

Writes results/<site>/tables/lucas_agreement.csv (+ per-class table).

Usage:
    python scripts/08_lucas.py --site slovenia
    python scripts/08_lucas.py --site ljubljana
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.warp import transform as warp_xy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lulckit.composite import PLANETARY  # noqa: E402
from lulckit.grid import aoi  # noqa: E402
from lulckit.labels import to_scheme  # noqa: E402
from lulckit.sites import CACHE, CLASSES, site  # noqa: E402

LUCAS_MAP = {"A": 2, "B": 3, "C": 4, "D": 5, "E": 5, "F": 6, "G": 1, "H": 5}


def lucas(year):
    if year == 2018:
        g = gpd.read_file(CACHE / "shared" / "lucas2018_SI.gpkg")
        g["lc"] = g["LC1"].astype(str)
        pts = g.to_crs(4326).geometry.representative_point()
    else:
        g = gpd.read_file(CACHE / "shared" / "lucas2022_SI.gpkg")
        g["lc"] = g["survey_lc1"].astype(str)
        pts = gpd.points_from_xy(g.point_long, g.point_lat, crs=4326)
    df = pd.DataFrame({"lon": [p.x for p in pts], "lat": [p.y for p in pts],
                       "ref": g["lc"].str[0].map(LUCAS_MAP)})
    return df.dropna().astype({"ref": int}).reset_index(drop=True)


def sample_product(collection, year, df):
    """Value of a PC-hosted 10 m product at each point (native resolution)."""
    cat = Client.open(PLANETARY, modifier=pc.sign_inplace)
    bbox = [df.lon.min() - 0.01, df.lat.min() - 0.01, df.lon.max() + 0.01, df.lat.max() + 0.01]
    dt = f"{year}-01-01/{year}-01-02" if collection.startswith("io") else f"{year}-01-01/{year}-12-31"
    items = list(cat.search(collections=[collection], bbox=bbox, datetime=dt).items())
    band = "map" if collection == "esa-worldcover" else "data"
    out = np.zeros(len(df), dtype=int)
    for it in items:
        with rasterio.open(it.assets[band].href) as r:
            xs, ys = warp_xy("EPSG:4326", r.crs, df.lon.tolist(), df.lat.tolist())
            b = r.bounds
            inside = [(b.left <= x < b.right and b.bottom < y <= b.top) for x, y in zip(xs, ys)]
            idx = np.flatnonzero(inside)
            if len(idx) == 0:
                continue
            vals = np.array([v[0] for v in r.sample([(xs[i], ys[i]) for i in idx])])
            v = to_scheme(vals, collection)
            out[idx] = np.where(out[idx] == 0, v, out[idx])
    return out


def sample_raster(path, df):
    with rasterio.open(path) as r:
        xs, ys = warp_xy("EPSG:4326", r.crs, df.lon.tolist(), df.lat.tolist())
        return np.array([v[0] for v in r.sample(list(zip(xs, ys)))])


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def score(name, ref, pred, year_ref):
    ok = pred > 0
    r, p = ref[ok], pred[ok]
    k = int(np.sum(r == p))
    lo, hi = wilson(k, len(r))
    rows = [dict(product=name, lucas_year=year_ref, n=len(r), OA=k / max(1, len(r)),
                 OA_lo=lo, OA_hi=hi)]
    pcs = []
    for c in sorted(set(r) | set(p)):
        tp = np.sum((r == c) & (p == c))
        pcs.append(dict(product=name, lucas_year=year_ref, class_id=c, name=CLASSES[c],
                        n_ref=int(np.sum(r == c)), n_map=int(np.sum(p == c)),
                        PA=tp / max(1, np.sum(r == c)), UA=tp / max(1, np.sum(p == c))))
    return rows, pcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    a = ap.parse_args()
    s = site(a.site)
    poly = aoi(s).to_crs(4326).geometry.iloc[0]
    rows, pcs, pts = [], [], []
    for ly in (2018, 2022):
        df = lucas(ly)
        inside = gpd.points_from_xy(df.lon, df.lat, crs=4326).within(poly)
        df = df[inside].reset_index(drop=True)
        prods = {}
        if s["name"] == "slovenia":
            prods["CORINE 2018 (100 m)"] = sample_raster(s["cache"] / "map_clc_2018.tif", df)
        else:
            cfg = json.load(open(s["tables"] / "best_config.json"))
            for y in s["years"]:
                prods[f"This study, S2 {y}"] = sample_raster(
                    s["cache"] / f"map10_{y}_{cfg['model']}.tif", df)
        prods[f"IO {min(ly, 2023)} (10 m)"] = sample_product("io-lulc-annual-v02", min(ly, 2023), df)
        prods["WorldCover 2021 (10 m)"] = sample_product("esa-worldcover", 2021, df)
        for name, pred in prods.items():
            r, pc_ = score(name, df.ref.to_numpy(), pred, ly)
            rows += r
            pcs += pc_
        d = df.copy()
        d["lucas_year"] = ly
        for name, pred in prods.items():
            d[name] = pred
        pts.append(d)
        print(f"LUCAS {ly}: {len(df)} points, reference classes "
              f"{df.ref.map(CLASSES).value_counts().to_dict()}")
    t = pd.DataFrame(rows)
    t.to_csv(s["tables"] / "lucas_agreement.csv", index=False, float_format="%.4f")
    pd.DataFrame(pcs).to_csv(s["tables"] / "lucas_perclass.csv", index=False, float_format="%.4f")
    pd.concat(pts).to_csv(s["data"] / "lucas_points.csv", index=False, float_format="%.6f")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
