"""Reference land-cover products and the stable-consensus training sample.

No field data or hand-digitised polygons exist for these sites, so labels come
from two independent global 10 m products, kept only where they agree:

* ESA WorldCover v200 (2021) - Zanaga et al.
* Impact Observatory / Esri annual LULC v02 (2017-2023) - Karra et al.

A pixel enters the sampling frame only if WorldCover 2021 and the IO maps for
every epoch used (2018, 2021, 2023) all give it the same harmonised class, and
its 3 x 3 neighbourhood is that class too. That restricts training to
unambiguous, unchanging pixels. It also means the hold-out accuracy measures
how well a classifier recovers *easy* pixels; it is an upper bound on map
accuracy, not an estimate of it. The papers say so.
"""

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer as pc
from pystac_client import Client
from scipy import ndimage

from .composite import PLANETARY, _bbox_ll
from .sites import CCI_MAP, IO_MAP, WORLDCOVER_MAP


def _lut(mapping, size=256):
    lut = np.zeros(size, dtype="uint8")
    for k, v in mapping.items():
        lut[k] = v
    return lut


def load_product(gbox, collection, year, band="data", resampling="mode"):
    """Read one year of a PC-hosted categorical product onto ``gbox``."""
    cat = Client.open(PLANETARY, modifier=pc.sign_inplace)
    if collection == "esa-worldcover":
        dt = f"{year}-01-01/{year}-12-31"
        band = "map"
    elif collection == "io-lulc-annual-v02":
        dt = f"{year}-01-01/{year}-01-02"
    elif collection == "esa-cci-lc":
        dt = f"{year}-01-01/{year}-12-31"
        band = "lccs_class"
    else:
        raise ValueError(collection)
    items = list(cat.search(collections=[collection], bbox=_bbox_ll(gbox), datetime=dt).items())
    if not items:
        raise RuntimeError(f"{collection} {year}: no items")
    ds = odc.stac.load(items, bands=[band], geobox=gbox, resampling=resampling,
                       chunks={"x": 2048, "y": 2048}, groupby="solar_day")
    arr = ds[band].max("time").values if "time" in ds[band].dims else ds[band].values
    return arr


def to_scheme(arr, collection):
    m = {"esa-worldcover": WORLDCOVER_MAP, "io-lulc-annual-v02": IO_MAP,
         "esa-cci-lc": CCI_MAP}[collection]
    return _lut(m)[arr.astype("int64").clip(0, 255)]


def consensus(maps, mask):
    """Class where every map agrees and the 3x3 neighbourhood is homogeneous."""
    base = maps[0].copy()
    agree = mask & (base > 0)
    for m in maps[1:]:
        agree &= m == base
    c = np.where(agree, base, 0).astype("uint8")
    homog = (ndimage.minimum_filter(c, 3) == c) & (ndimage.maximum_filter(c, 3) == c)
    return np.where(homog, c, 0).astype("uint8")


def draw_sample(cons, gbox, n_per_class, block_m, seed=42, min_sep_px=3):
    """Stratified random pixels from the consensus raster, with spatial blocks.

    ``min_sep_px`` thins candidates to a lattice so no two samples are
    adjacent pixels (adjacent pixels are near-duplicates).
    """
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    lat = np.zeros_like(cons, dtype=bool)
    lat[::min_sep_px, ::min_sep_px] = True
    out = []
    for k in np.unique(cons):
        if k == 0:
            continue
        r, c = np.nonzero((cons == k) & lat)
        n = min(n_per_class, len(r))
        pick = rng.choice(len(r), n, replace=False)
        out.append(pd.DataFrame({"row": r[pick], "col": c[pick], "y": k}))
    df = pd.concat(out, ignore_index=True)
    a = gbox.affine
    df["x_m"] = a.c + (df["col"] + 0.5) * a.a
    df["y_m"] = a.f + (df["row"] + 0.5) * a.e
    bx = np.floor(df["x_m"] / block_m).astype(int)
    by = np.floor(df["y_m"] / block_m).astype(int)
    df["block"] = pd.factorize(bx.astype(str) + "_" + by.astype(str))[0]
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
