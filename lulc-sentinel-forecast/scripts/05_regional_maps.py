"""Step 5 (region sites): harmonised historical land-cover maps.

    Uttar Pradesh  ESA CCI Land Cover, 300 m, 1995-2020 (5-year steps)
    Slovenia       CORINE Land Cover, rasterised at 100 m, 1990-2018

Plus, for both, the Sentinel-2-based Impact Observatory 10 m annual maps for
2017 and 2023, mode-aggregated onto the same grid. These are not used to fit
the forecast; they are the Sentinel-era check on what the coarse long-record
products say about recent change.

Writes data/cache/<site>/map_<source>_<year>.tif and
results/<site>/tables/class_areas_history.csv.

Usage:
    python scripts/05_regional_maps.py --site uttar_pradesh
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lulckit import labels as L  # noqa: E402
from lulckit.grid import aoi_mask, geobox, read_tif, write_tif  # noqa: E402
from lulckit.sites import CLASSES, clc_to_scheme, site  # noqa: E402

CLC_URL = "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC{y}_WM/MapServer/0/query"
CLC_FIELD = {1990: "code_90", 2000: "code_00", 2006: "Code_06", 2012: "Code_12", 2018: "Code_18"}


def _get(url, data=None, tries=6):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except Exception as e:  # transient server errors
            print(f"    retry {attempt}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(url[:120])


def fetch_clc(year, bbox_ll, cache):
    """All CORINE polygons intersecting the bbox.

    The service ignores resultOffset for GeoJSON output, so the object IDs are
    listed first and features are requested by ID in chunks.
    """
    out = cache / f"clc_{year}.gpkg"
    if out.exists():
        return gpd.read_file(out)
    base = CLC_URL.format(y=year)
    q = dict(geometry=",".join(map(str, bbox_ll)), geometryType="esriGeometryEnvelope",
             inSR=4326, spatialRel="esriSpatialRelIntersects", returnIdsOnly="true", f="json")
    ids = sorted(_get(base + "?" + urllib.parse.urlencode(q))["objectIds"])
    def chunk(sub, offset=None):
        """Fetch IDs; on a server error bisect, then generalise (~1 m) a lone polygon."""
        q = dict(objectIds=",".join(map(str, sub)), outFields=CLC_FIELD[year],
                 returnGeometry="true", outSR=4326, f="geojson")
        if offset:
            q["maxAllowableOffset"] = offset
        try:
            return _get(base + "?" + urllib.parse.urlencode(q), tries=2)["features"]
        except RuntimeError:
            if len(sub) > 1:
                h = len(sub) // 2
                return chunk(sub[:h]) + chunk(sub[h:])
            # A lone polygon the server cannot serve: generalise in steps up
            # to ~40 m, still well inside one 100 m output cell.
            ladder = [None, 0.00001, 0.0001, 0.0005]
            nxt = ladder.index(offset) + 1
            if nxt < len(ladder):
                print(f"    object {sub[0]}: generalising geometry ({ladder[nxt]} deg)")
                return chunk(sub, offset=ladder[nxt])
            raise

    feats = []
    for i in range(0, len(ids), 50):
        feats += chunk(ids[i:i + 50])
    g = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    g = g.rename(columns={CLC_FIELD[year]: "code"})
    g["code"] = g["code"].astype(int)
    assert len(g) == len(ids), (len(g), len(ids))
    g.to_file(out, driver="GPKG")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    a = ap.parse_args()
    s = site(a.site)
    gb = geobox(s, s["res"], pad=s["res"] * 3)
    mask = aoi_mask(s, gb)
    np.save(s["cache"] / "mask.npy", mask)
    print(f"{s['name']}: grid {gb.shape} at {s['res']} m, {mask.sum()} cells in AOI")

    maps = {}
    if s["source"] == "corine":
        bb = gb.geographic_extent.boundingbox
        bbox = [bb.left, bb.bottom, bb.right, bb.top]
        for y in [1990] + s["epochs"]:
            p = s["cache"] / f"map_clc_{y}.tif"
            if not p.exists():
                g = fetch_clc(y, bbox, s["cache"]).to_crs(gb.crs)
                g["k"] = g["code"].map(clc_to_scheme)
                r = rasterize(zip(g.geometry, g["k"]), out_shape=gb.shape,
                              transform=gb.affine, fill=0, dtype="uint8")
                r[~mask] = 0
                write_tif(p, r, gb, nodata=0)
                print(f"  CLC {y}: {len(g)} polygons")
            maps[("CLC", y)] = read_tif(p)[0]
    else:
        years = sorted(set(s["epochs"]) | {2000, 2010})
        for y in years:
            p = s["cache"] / f"map_cci_{y}.tif"
            if not p.exists():
                raw = L.load_product(gb, "esa-cci-lc", y, resampling="mode")
                r = L.to_scheme(raw, "esa-cci-lc")
                r[~mask] = 0
                write_tif(p, r, gb, nodata=0)
                print(f"  CCI {y}")
            maps[("CCI", y)] = read_tif(p)[0]

    for y in (2017, 2020, 2023):
        p = s["cache"] / f"map_io_{y}.tif"
        if not p.exists():
            t = time.time()
            raw = L.load_product(gb, "io-lulc-annual-v02", y, resampling="mode")
            r = L.to_scheme(raw, "io-lulc-annual-v02")
            r[~mask] = 0
            write_tif(p, r, gb, nodata=0)
            print(f"  IO {y} ({time.time() - t:.0f} s)")
        maps[("IO", y)] = read_tif(p)[0]

    # ESA WorldCover 2021: a second, independent Sentinel-era estimate, because
    # IO and the long-record products disagree strongly on built-up area.
    p = s["cache"] / "map_wc_2021.tif"
    if not p.exists():
        raw = L.load_product(gb, "esa-worldcover", 2021, resampling="mode")
        r = L.to_scheme(raw, "esa-worldcover")
        r[~mask] = 0
        write_tif(p, r, gb, nodata=0)
        print("  WorldCover 2021")
    maps[("WorldCover", 2021)] = read_tif(p)[0]

    cell_km2 = (s["res"] / 1000) ** 2
    rows = []
    for (src, y), m in sorted(maps.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        for k, n in CLASSES.items():
            rows.append(dict(source=src, year=y, class_id=k, name=n,
                             area_km2=float(np.sum(m[mask] == k) * cell_km2),
                             share=float(np.mean(m[mask] == k))))
    tab = pd.DataFrame(rows)
    tab.to_csv(s["tables"] / "class_areas_history.csv", index=False, float_format="%.4f")
    print(tab.pivot_table(index=["source", "year"], columns="name", values="share").round(4))


if __name__ == "__main__":
    main()
