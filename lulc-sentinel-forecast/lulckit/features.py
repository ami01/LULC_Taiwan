"""Feature stacks from the seasonal composites, at sample points or wall-to-wall.

Feature names are ``<sensor>_<season>_<band>``, e.g. ``s2_rabi_B8`` or
``s1_summer_VH``. Indices are computed from the stored reflectances.
"""

import numpy as np
import rasterio

from .composite import indices

S2_10M = ["B2", "B3", "B4", "B8"]
S2_ALL = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
L8_ALL = ["blue", "green", "red", "nir", "swir1", "swir2"]
IDX_S2 = ["NDVI", "NDBI", "MNDWI", "BSI", "NDRE"]
IDX_L8 = ["NDVI", "NDBI", "MNDWI", "BSI"]
S1_ALL = ["VV", "VH", "VVminusVH"]


def feature_sets(s):
    """The experimental design: which inputs each feature set uses."""
    seas = list(s["seasons"])
    one = s["single_season"]
    f = lambda sensor, season, bands: [f"{sensor}_{season}_{b}" for b in bands]
    multi = lambda sensor, bands: sum((f(sensor, x, bands) for x in seas), [])
    return {
        "S2-4b-1s": f("s2", one, S2_10M),
        "S2-10b-1s": f("s2", one, S2_ALL),
        "S2-10b+idx-1s": f("s2", one, S2_ALL + IDX_S2),
        "S2-multiseason": multi("s2", S2_ALL + IDX_S2),
        "S2+S1-multiseason": multi("s2", S2_ALL + IDX_S2) + multi("s1", S1_ALL),
        "S1-multiseason": multi("s1", S1_ALL),
        "L8-1s": f("l8", one, L8_ALL + IDX_L8),
        "L8-multiseason": multi("l8", L8_ALL + IDX_L8),
    }


def _read(path, rows=None, cols=None, xs=None, ys=None):
    with rasterio.open(path) as r:
        names = list(r.descriptions)
        if xs is not None:
            rr, cc = rasterio.transform.rowcol(r.transform, xs, ys)
            rows, cols = np.asarray(rr), np.asarray(cc)
        a = r.read()
        nod = r.nodata
    v = a[:, rows, cols].astype("float32") if rows is not None else a.astype("float32")
    v[v == nod] = np.nan
    return names, v


def _expand(sensor, season, names, v):
    """Scale stored integers back to physical units, add indices."""
    scale = 100.0 if sensor == "s1" else 10000.0
    d = {n: v[i] / scale for i, n in enumerate(names)}
    if sensor != "s1":
        d.update(indices(d))
    return {f"{sensor}_{season}_{k}": x for k, x in d.items()}


def at_points(s, year, df, sensors=("s2", "s1", "l8")):
    """Dict of feature name -> 1-D array for the sample points in ``df``."""
    out = {}
    for sensor in sensors:
        for season in s["seasons"]:
            p = s["cache"] / f"{sensor}_{year}_{season}.tif"
            if not p.exists():
                continue
            if sensor == "l8":
                names, v = _read(p, xs=df["x_m"].values, ys=df["y_m"].values)
            else:
                names, v = _read(p, rows=df["row"].values, cols=df["col"].values)
            out.update(_expand(sensor, season, names, v))
    return out


def wall_to_wall(s, year, wanted, row_slice=None):
    """Feature cube [n_features, h, w] in the order of ``wanted`` (10 m sensors)."""
    need = {}
    for f in wanted:
        sensor, season, _ = f.split("_", 2)
        need.setdefault((sensor, season), []).append(f)
    planes = {}
    for (sensor, season), fs in need.items():
        p = s["cache"] / f"{sensor}_{year}_{season}.tif"
        with rasterio.open(p) as r:
            names = list(r.descriptions)
            win = None
            if row_slice is not None:
                win = rasterio.windows.Window(0, row_slice.start, r.width,
                                              row_slice.stop - row_slice.start)
            a = r.read(window=win).astype("float32")
            a[a == r.nodata] = np.nan
        planes.update(_expand(sensor, season, names, a))
    return np.stack([planes[f] for f in wanted])
