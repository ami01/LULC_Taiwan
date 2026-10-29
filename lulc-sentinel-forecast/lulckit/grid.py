"""Study-area geometry, analysis grids and GeoTIFF I/O."""

import geopandas as gpd
import numpy as np
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.features import rasterize

from .sites import RAW


def aoi(s) -> gpd.GeoDataFrame:
    """Study-area polygon in the site CRS."""
    fname, col, val = s["boundary"]
    g = gpd.read_file(RAW / "boundaries" / fname)
    if col is not None:
        g = g[g[col] == val]
    if len(g) != 1:
        raise ValueError(f"boundary lookup for {s['name']} matched {len(g)} rows")
    return g.to_crs(s["crs"])[["geometry"]].reset_index(drop=True)


def geobox(s, res: float, pad: float = 0.0) -> GeoBox:
    """North-up grid covering the AOI, snapped to multiples of ``res``."""
    xmin, ymin, xmax, ymax = aoi(s).total_bounds
    xmin = np.floor((xmin - pad) / res) * res
    ymin = np.floor((ymin - pad) / res) * res
    xmax = np.ceil((xmax + pad) / res) * res
    ymax = np.ceil((ymax + pad) / res) * res
    return GeoBox.from_bbox((xmin, ymin, xmax, ymax), crs=s["crs"], resolution=res)


def coarsen(gbox: GeoBox, f: int) -> GeoBox:
    """Grid whose cells are exact f x f blocks of ``gbox`` (remainder dropped)."""
    from affine import Affine
    h, w = gbox.shape
    return GeoBox((h // f, w // f), gbox.affine * Affine.scale(f), gbox.crs)


def aoi_mask(s, gbox: GeoBox) -> np.ndarray:
    return rasterize([(g, 1) for g in aoi(s).geometry], out_shape=gbox.shape,
                     transform=gbox.affine, fill=0, dtype="uint8",
                     all_touched=False).astype(bool)


def write_tif(path, arr, gbox: GeoBox, nodata=None, names=None):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    profile = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                   count=arr.shape[0], dtype=arr.dtype, crs=str(gbox.crs),
                   transform=gbox.affine, compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256, nodata=nodata)
    if arr.dtype.kind == "f":
        profile["predictor"] = 3
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        if names:
            for i, n in enumerate(names, 1):
                dst.set_band_description(i, n)


def read_tif(path):
    """Return (array, GeoBox, band names)."""
    with rasterio.open(path) as r:
        a = r.read()
        gb = GeoBox(r.shape, r.transform, r.crs)
        names = list(r.descriptions)
    return (a[0] if a.shape[0] == 1 else a), gb, names


def mode_downsample(a: np.ndarray, f: int, nodata=0) -> np.ndarray:
    """Majority-resample a categorical raster by an integer factor."""
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    b = a[:h, :w].reshape(h // f, f, w // f, f).transpose(0, 2, 1, 3).reshape(h // f, w // f, f * f)
    classes = [c for c in np.unique(b) if c != nodata]
    counts = np.stack([(b == c).sum(-1) for c in classes], -1)
    out = np.asarray(classes, dtype=a.dtype)[counts.argmax(-1)]
    out[counts.max(-1) == 0] = nodata
    return out
