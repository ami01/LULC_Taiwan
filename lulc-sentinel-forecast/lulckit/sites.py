"""Study-site definitions.

Everything site-specific lives here. Two kinds of site:

* ``city``   - a Sentinel-era (2018-2024) 10 m study: classifier and
  feature-set comparison on Sentinel-2 / Sentinel-1 / Landsat, maps for three
  epochs, and a 30 m land-change forecast. Varanasi and Ljubljana.
* ``region`` - a long-record study on an existing land-cover series
  (ESA CCI 300 m for Uttar Pradesh, CORINE 100 m for Slovenia), with a
  Sentinel-era cross-check against the 10 m Impact Observatory series.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"          # large downloads, not committed
RESULTS = ROOT / "results"

# --------------------------------------------------------------------------
# Harmonised class scheme, shared by all four studies
# --------------------------------------------------------------------------
CLASSES = {
    1: "Water",
    2: "Built-up",
    3: "Cropland",
    4: "Tree cover",
    5: "Grass/shrub",
    6: "Bare",
}
# Land-cover hues kept conventional (water blue, built red, crop gold, trees
# dark green, grass light green, bare brown) but stepped to pass the dataviz
# palette validator: lightness band, chroma floor, adjacent CVD dE >= 8.
CLASS_COLORS = {
    1: "#2a68c6",
    2: "#d7301f",
    3: "#c9971a",
    4: "#1b7837",
    5: "#6aad3d",
    6: "#a8642a",
}

# ESA WorldCover v200 -> scheme
WORLDCOVER_MAP = {10: 4, 20: 5, 30: 5, 40: 3, 50: 2, 60: 6, 70: 6, 80: 1,
                  90: 5, 95: 4, 100: 5}
# Impact Observatory / Esri annual LULC v02 -> scheme (10 = cloud -> nodata)
IO_MAP = {1: 1, 2: 4, 4: 5, 5: 3, 7: 2, 8: 6, 9: 6, 11: 5}
# ESA CCI LC -> scheme
CCI_MAP = {10: 3, 11: 3, 12: 3, 20: 3, 30: 3, 40: 5,
           50: 4, 60: 4, 61: 4, 62: 4, 70: 4, 71: 4, 72: 4, 80: 4, 81: 4,
           82: 4, 90: 4, 100: 4, 110: 5, 120: 5, 121: 5, 122: 5, 130: 5,
           140: 5, 150: 5, 151: 5, 152: 5, 153: 5, 160: 4, 170: 4, 180: 5,
           190: 2, 200: 6, 201: 6, 202: 6, 210: 1, 220: 6}


def clc_to_scheme(code: int) -> int:
    """CORINE Land Cover level-3 code -> scheme."""
    if code < 200:
        return 2                       # artificial surfaces
    if code == 231:
        return 5                       # pastures are grassland
    if code < 300:
        return 3                       # arable, permanent crops, heterogeneous
    if code < 320:
        return 4                       # forests
    if code < 330:
        return 5                       # grassland, heath, transitional shrub
    if code < 400:
        return 6                       # bare rock, sparse, burnt, glaciers
    if code < 500:
        return 5                       # wetlands
    return 1                           # water


SITES = {
    "varanasi": dict(
        kind="city",
        title="Varanasi district, Uttar Pradesh, India",
        boundary=("IND_ADM2.geojson", "shapeName", "Varanasi"),
        crs="EPSG:32644",
        years=[2018, 2021, 2024],
        ref_year=2021,
        # Three cloud-light windows of the Gangetic plain year. The monsoon
        # (mid-June to September) is left to Sentinel-1.
        seasons={
            "rabi": ("01-01", "03-15"),        # winter crops green
            "premonsoon": ("03-16", "06-10"),  # harvested, fields bare
            "postmonsoon": ("10-01", "12-31"),  # kharif harvest, sowing
        },
        single_season="postmonsoon",
        osm=("https://download.openstreetmap.fr/extracts/asia/india/"
             "uttar_pradesh-latest.osm.pbf"),
        block_km=2.0,
        forecast_res=30,
    ),
    "ljubljana": dict(
        kind="city",
        title="City Municipality of Ljubljana, Slovenia",
        boundary=("SVN_ADM2.geojson", "shapeName", "Ljubljana"),
        crs="EPSG:32633",
        years=[2018, 2021, 2024],
        ref_year=2021,
        seasons={
            "spring": ("03-20", "05-31"),
            "summer": ("06-01", "08-31"),
            "autumn": ("09-01", "10-31"),
        },
        single_season="summer",
        osm="https://download.bbbike.org/osm/bbbike/Ljubljana/Ljubljana.osm.pbf",
        block_km=1.0,
        forecast_res=30,
    ),
    "uttar_pradesh": dict(
        kind="region",
        title="Uttar Pradesh, India",
        boundary=("IND_ADM1.geojson", "shapeName", "Uttar Pradesh"),
        districts=("IND_ADM2.geojson", "shapeName"),
        crs="EPSG:32644",
        res=300,
        source="esa-cci-lc",
        epochs=[1995, 2005, 2015, 2020],
        osm=("https://download.openstreetmap.fr/extracts/asia/india/"
             "uttar_pradesh-latest.osm.pbf"),
    ),
    "slovenia": dict(
        kind="region",
        title="Slovenia",
        boundary=("SVN_ADM0.geojson", None, None),
        districts=("SVN_ADM2.geojson", "shapeName"),
        crs="EPSG:3035",
        res=100,
        source="corine",
        epochs=[2000, 2006, 2012, 2018],
        roads="grip4",
    ),
}


def site(name: str) -> dict:
    s = dict(SITES[name])
    s["name"] = name
    s["cache"] = CACHE / name
    s["out"] = RESULTS / name
    s["tables"] = s["out"] / "tables"
    s["figures"] = s["out"] / "figures"
    s["data"] = DATA / name
    for k in ("cache", "out", "tables", "figures", "data"):
        s[k].mkdir(parents=True, exist_ok=True)
    return s
