"""Seasonal median composites from Sentinel-2 L2A, Sentinel-1 RTC and Landsat 8/9.

All three are read straight from cloud-optimised GeoTIFFs through STAC:

* Sentinel-2 L2A  - Earth Search (AWS), collection ``sentinel-2-l2a``
* Sentinel-1 RTC  - Microsoft Planetary Computer, ``sentinel-1-rtc``
* Landsat C2 L2   - Microsoft Planetary Computer, ``landsat-c2-l2``

Sentinel-2 note. From processing baseline 04.00 (25 January 2022) ESA added a
+1000 offset to L2A digital numbers. Earth Search flags those scenes with
``earthsearch:boa_offset_applied = True`` and also declares ``offset = -0.1`` on
each band, which reads as if the offset were still in the pixels. It is not:
checked on the same acquisition (S2A_33TVL_20210729) at baselines 03.01 and
05.00, the median blue DN is 294 and 306. The +1000 has already been removed
from the COGs, so no offset is subtracted here. Subtracting it - the obvious
reading of the metadata - drives blue and red to zero over vegetation.
"""

import dask
import numpy as np
import odc.stac
import planetary_computer as pc
import xarray as xr
from pystac_client import Client

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1"

S2_BANDS = {"B2": "blue", "B3": "green", "B4": "red", "B5": "rededge1",
            "B6": "rededge2", "B7": "rededge3", "B8": "nir", "B8A": "nir08",
            "B11": "swir16", "B12": "swir22"}
# Scene classification: keep vegetation, bare, water, unclassified, snow.
SCL_CLEAR = [4, 5, 6, 7, 11]

L8_BANDS = {"blue": "blue", "green": "green", "red": "red", "nir": "nir08",
            "swir1": "swir16", "swir2": "swir22"}

# 4 workers x (768^2 px x ~12 dates x 11 bands) keeps a 24 Mpx AOI under ~4 GB.
dask.config.set(scheduler="threads", num_workers=4)


def _bbox_ll(gbox):
    b = gbox.geographic_extent.boundingbox
    return [b.left, b.bottom, b.right, b.top]


def _pick(items, max_dates, cc_key="eo:cloud_cover"):
    """Least-cloudy scenes, at most ``max_dates`` distinct acquisition days."""
    items = sorted(items, key=lambda i: i.properties.get(cc_key, 0))
    days, keep = set(), []
    for it in items:
        d = it.datetime.date()
        if d in days or len(days) < max_dates:
            days.add(d)
            keep.append(it)
    return keep


def s2_items(gbox, start, end, max_cloud=60, max_dates=12):
    cat = Client.open(EARTH_SEARCH)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=_bbox_ll(gbox),
                            datetime=f"{start}/{end}",
                            query={"eo:cloud_cover": {"lt": max_cloud}}).items())
    # A handful of items reference only the requester-pays JP2 bucket; skip them.
    items = [i for i in items if i.assets["red"].href.startswith("https://")]
    return _pick(items, max_dates)


def s2_composite(gbox, start, end, max_cloud=60, max_dates=12):
    """Median surface reflectance (float32, 0-1) of cloud-free observations.

    Returns (DataArray[band, y, x], n_scenes, clear-observation count).
    """
    items = s2_items(gbox, start, end, max_cloud, max_dates)
    if not items:
        raise RuntimeError(f"no Sentinel-2 scenes {start}..{end}")
    ds = odc.stac.load(items, bands=list(S2_BANDS.values()) + ["scl"],
                       geobox=gbox, groupby="solar_day",
                       chunks={"x": 768, "y": 768}, resampling="nearest")
    clear = ds["scl"].isin(SCL_CLEAR)
    out = []
    for b in S2_BANDS.values():
        v = ds[b].astype("float32")
        v = v.where((v > 0) & clear)
        out.append((v / 10000.0).clip(0, 1.5))
    stack = xr.concat(out, dim="band").assign_coords(band=list(S2_BANDS))
    stack = stack.chunk({"time": -1})
    med = stack.median("time", skipna=True)
    nobs = clear.sum("time")
    med, nobs = dask.compute(med, nobs)
    return med.astype("float32"), len({i.datetime.date() for i in items}), nobs


def s1_composite(gbox, start, end, max_dates=15):
    """Median VV and VH backscatter (gamma0, dB) from Sentinel-1 RTC."""
    cat = Client.open(PLANETARY, modifier=pc.sign_inplace)
    items = list(cat.search(collections=["sentinel-1-rtc"], bbox=_bbox_ll(gbox),
                            datetime=f"{start}/{end}").items())
    if not items:
        raise RuntimeError(f"no Sentinel-1 scenes {start}..{end}")
    # Spread over the window rather than the first N.
    items = sorted(items, key=lambda i: i.datetime)
    if len(items) > max_dates * 2:
        idx = np.linspace(0, len(items) - 1, max_dates * 2).round().astype(int)
        items = [items[i] for i in sorted(set(idx))]
    ds = odc.stac.load(items, bands=["vv", "vh"], geobox=gbox, groupby="solar_day",
                       chunks={"x": 768, "y": 768}, resampling="bilinear")
    ds = ds.where(ds > 0).chunk({"time": -1})
    med = ds.median("time", skipna=True)
    med = dask.compute(med)[0]
    vv = 10 * np.log10(med["vv"].clip(1e-5))
    vh = 10 * np.log10(med["vh"].clip(1e-5))
    da = xr.concat([vv, vh, vv - vh], dim="band").assign_coords(band=["VV", "VH", "VVminusVH"])
    return da.astype("float32"), len({i.datetime.date() for i in items})


def landsat_composite(gbox, start, end, max_cloud=60, max_dates=12):
    """Median Landsat 8/9 Collection-2 Level-2 surface reflectance at 30 m."""
    cat = Client.open(PLANETARY, modifier=pc.sign_inplace)
    items = list(cat.search(collections=["landsat-c2-l2"], bbox=_bbox_ll(gbox),
                            datetime=f"{start}/{end}",
                            query={"platform": {"in": ["landsat-8", "landsat-9"]},
                                   "eo:cloud_cover": {"lt": max_cloud}}).items())
    items = _pick(items, max_dates)
    if not items:
        raise RuntimeError(f"no Landsat scenes {start}..{end}")
    ds = odc.stac.load(items, bands=list(L8_BANDS.values()) + ["qa_pixel"],
                       geobox=gbox, groupby="solar_day",
                       chunks={"x": 768, "y": 768}, resampling="nearest")
    qa = ds["qa_pixel"]
    # bits: 0 fill, 1 dilated cloud, 3 cloud, 4 cloud shadow
    bad = ((qa & 0b11011) > 0) | (qa == 0)
    out = []
    for b in L8_BANDS.values():
        v = ds[b].astype("float32")
        v = v.where((v > 0) & ~bad) * 0.0000275 - 0.2
        out.append(v.clip(0, 1.5))
    stack = xr.concat(out, dim="band").assign_coords(band=list(L8_BANDS)).chunk({"time": -1})
    med = dask.compute(stack.median("time", skipna=True))[0]
    return med.astype("float32"), len({i.datetime.date() for i in items})


def indices(b: dict) -> dict:
    """Spectral indices from a dict of reflectance arrays.

    Accepts Sentinel-2 names (B2..B12) or Landsat names (blue..swir2).
    """
    g = lambda *ks: next(b[k] for k in ks if k in b)
    blue, green, red = g("B2", "blue"), g("B3", "green"), g("B4", "red")
    nir, sw1, sw2 = g("B8", "nir"), g("B11", "swir1"), g("B12", "swir2")
    eps = 1e-6
    out = {
        "NDVI": (nir - red) / (nir + red + eps),
        "NDBI": (sw1 - nir) / (sw1 + nir + eps),
        "MNDWI": (green - sw1) / (green + sw1 + eps),
        "BSI": ((sw1 + red) - (nir + blue)) / ((sw1 + red) + (nir + blue) + eps),
    }
    if "B5" in b:
        out["NDRE"] = (b["B8A"] - b["B5"]) / (b["B8A"] + b["B5"] + eps)
    return out
