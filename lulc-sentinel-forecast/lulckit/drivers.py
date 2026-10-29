"""Driver (explanatory) variables for the transition-potential models.

    elevation, slope          Copernicus GLO-30 DEM (Planetary Computer)
    dist_major_road           motorway/trunk/primary (OSM) or GRIP4 types 1-2
    dist_minor_road           secondary/tertiary (OSM) or GRIP4 types 3-4
    dist_rail                 OSM railway=rail (not available for Slovenia)
    dist_water                JRC Global Surface Water occurrence >= 50 %
    dist_centre               OSM place=city|town, or GeoNames seats for Slovenia
    log_pop                   log(1 + WorldPop 1 km persons per km2), year-matched

Road, rail and settlement layers are present-day snapshots used for every
period. That is the usual compromise in land-change modelling and it flatters
hindcasts slightly: a road built in 2012 'explains' growth along it in 2005.
"""

import zipfile

import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from shapely import wkb

from .composite import PLANETARY, _bbox_ll
from .sites import CACHE

SHARED = CACHE / "shared"
MAJOR = {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}
MINOR = {"secondary", "tertiary", "secondary_link", "tertiary_link"}
STATIC = ["elevation", "slope", "dist_major_road", "dist_minor_road", "dist_rail",
          "dist_water", "dist_centre", "log_pop"]
THREE_VAR = ["elevation", "slope", "dist_major_road"]


def osm_layers(pbf, out_gpkg):
    """Roads, rail, rivers and settlement nodes from an OSM .pbf (cached)."""
    if out_gpkg.exists():
        return (gpd.read_file(out_gpkg, layer="lines"),
                gpd.read_file(out_gpkg, layer="places"))
    import osmium

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.fab = osmium.geom.WKBFactory()
            self.lines, self.places = [], []

        def way(self, w):
            hw, rw = w.tags.get("highway"), w.tags.get("railway")
            cls = ("major" if hw in MAJOR else "minor" if hw in MINOR else
                   "rail" if rw == "rail" else None)
            if cls:
                try:
                    self.lines.append((cls, wkb.loads(self.fab.create_linestring(w), hex=True)))
                except Exception:
                    pass

        def node(self, n):
            p = n.tags.get("place")
            if p in ("city", "town"):
                self.places.append((p, n.tags.get("name"), n.location.lon, n.location.lat))

    h = H()
    h.apply_file(str(pbf), locations=True)
    lines = gpd.GeoDataFrame({"cls": [c for c, _ in h.lines]},
                             geometry=[g for _, g in h.lines], crs="EPSG:4326")
    pl = pd.DataFrame(h.places, columns=["place", "name", "lon", "lat"])
    places = gpd.GeoDataFrame(pl, geometry=gpd.points_from_xy(pl.lon, pl.lat), crs="EPSG:4326")
    lines.to_file(out_gpkg, layer="lines", driver="GPKG")
    places.to_file(out_gpkg, layer="places", driver="GPKG")
    return lines, places


def grip4_layers(gbox):
    """GRIP4 Europe roads clipped to the grid (Meijer et al. 2018)."""
    z = SHARED / "GRIP4_Region4_vector_shp.zip"
    with zipfile.ZipFile(z) as f:
        shp = [n for n in f.namelist() if n.endswith(".shp")][0]
    bb = gbox.geographic_extent.boundingbox
    g = gpd.read_file(f"zip://{z}!{shp}", bbox=(bb.left, bb.bottom, bb.right, bb.top))
    g["cls"] = np.where(g["GP_RTP"].isin([1, 2]), "major",
                        np.where(g["GP_RTP"].isin([3, 4]), "minor", "local"))
    return g[["cls", "geometry"]]


def geonames_places(min_pop=2000):
    """Slovenian municipal seats and towns from GeoNames."""
    cols = ["id", "name", "ascii", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2",
            "a1", "a2", "a3", "a4", "pop", "elev", "dem", "tz", "mod"]
    with zipfile.ZipFile(SHARED / "SI.zip") as f:
        df = pd.read_csv(f.open("SI.txt"), sep="\t", names=cols, low_memory=False)
    df = df[(df.fclass == "P") & ((df["pop"] >= min_pop) |
                                  df.fcode.isin(["PPLC", "PPLA", "PPLA2"]))]
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")


def _dist(geoms, gbox):
    if len(geoms) == 0:
        return np.full(gbox.shape, 1e6, dtype="float32")
    r = rasterize([(g, 1) for g in geoms], out_shape=gbox.shape, transform=gbox.affine,
                  fill=0, all_touched=True, dtype="uint8")
    if r.sum() == 0:
        return np.full(gbox.shape, 1e6, dtype="float32")
    return (ndimage.distance_transform_edt(r == 0) * abs(gbox.affine.a)).astype("float32")


def dem(gbox):
    cat = Client.open(PLANETARY, modifier=pc.sign_inplace)
    items = list(cat.search(collections=["cop-dem-glo-30"], bbox=_bbox_ll(gbox)).items())
    res = abs(gbox.affine.a)
    ds = odc.stac.load(items, bands=["data"], geobox=gbox,
                       resampling="bilinear" if res <= 30 else "average",
                       chunks={"x": 2048, "y": 2048})
    z = ds["data"].max("time").values.astype("float32")
    gy, gx = np.gradient(z, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    return z, slope


def water(gbox):
    cat = Client.open(PLANETARY, modifier=pc.sign_inplace)
    items = list(cat.search(collections=["jrc-gsw"], bbox=_bbox_ll(gbox)).items())
    ds = odc.stac.load(items, bands=["occurrence"], geobox=gbox, resampling="nearest",
                       chunks={"x": 2048, "y": 2048})
    occ = ds["occurrence"].max("time").values
    w = (occ >= 50) & (occ <= 100)
    if not w.any():
        return np.full(gbox.shape, 1e6, dtype="float32")
    return (ndimage.distance_transform_edt(~w) * abs(gbox.affine.a)).astype("float32")


def population(gbox, iso, year):
    year = min(max(year, 2000), 2020)
    avail = [2000, 2005, 2010, 2015, 2018, 2020]
    year = min(avail, key=lambda y: abs(y - year))
    src = SHARED / f"pop_{iso}_{year}.tif"
    out = np.zeros(gbox.shape, dtype="float32")
    with rasterio.open(src) as r:
        reproject(rasterio.band(r, 1), out, dst_transform=gbox.affine, dst_crs=str(gbox.crs),
                  resampling=Resampling.bilinear, src_nodata=r.nodata, dst_nodata=0)
    # persons per 1 km cell == persons per km2
    return np.log1p(np.clip(out, 0, None)).astype("float32"), year


def build(s, gbox, pop_year):
    """All drivers on ``gbox``. Static layers cached per grid, population per year."""
    res = int(abs(gbox.affine.a))
    cache = s["cache"] / f"drivers_{res}m.npz"
    if cache.exists():
        z = np.load(cache)
        d = {k: z[k] for k in z.files}
    else:
        d = {}
        d["elevation"], d["slope"] = dem(gbox)
        d["dist_water"] = water(gbox)
        if s.get("roads") == "grip4":
            roads = grip4_layers(gbox).to_crs(gbox.crs)
            places = geonames_places().to_crs(gbox.crs)
        else:
            pbf = SHARED / s["osm"].rsplit("/", 1)[1].replace("-latest", "")
            roads, places = osm_layers(pbf, s["cache"] / "osm_layers.gpkg")
            roads, places = roads.to_crs(gbox.crs), places.to_crs(gbox.crs)
            d["dist_rail"] = _dist(roads.geometry[roads.cls == "rail"], gbox)
        d["dist_major_road"] = _dist(roads.geometry[roads.cls == "major"], gbox)
        d["dist_minor_road"] = _dist(roads.geometry[roads.cls == "minor"], gbox)
        d["dist_centre"] = _dist(places.geometry, gbox)
        np.savez_compressed(cache, **d)
    iso = "IND" if s["crs"] == "EPSG:32644" else "SVN"
    d["log_pop"], _ = population(gbox, iso, pop_year)
    return d
