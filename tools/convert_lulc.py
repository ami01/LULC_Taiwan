"""Convert LULC GeoTIFFs into web-ready PNG overlays + metadata for the viewer.

Usage (from repo root):
    pip install rasterio numpy pillow
    python tools/convert_lulc.py

Reads tools/classes.json, writes docs/data/<year>.png and docs/data/lulc.js.
All layers are reprojected (nearest neighbour) onto one shared Web Mercator
grid so they line up exactly in the browser.
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data"
DST_CRS = "EPSG:3857"
NODATA = 0


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def main():
    cfg = json.loads((ROOT / "tools" / "classes.json").read_text())
    classes = {int(k): v for k, v in cfg["classes"].items()}
    OUT.mkdir(parents=True, exist_ok=True)

    # Shared target grid: union of all source extents.
    srcs = [rasterio.open(ROOT / l["file"]) for l in cfg["layers"]]
    ref = srcs[0]
    left = min(s.bounds.left for s in srcs); bottom = min(s.bounds.bottom for s in srcs)
    right = max(s.bounds.right for s in srcs); top = max(s.bounds.top for s in srcs)
    dst_t, dst_w, dst_h = calculate_default_transform(
        ref.crs, DST_CRS, ref.width, ref.height, left, bottom, right, top)
    west, north = dst_t * (0, 0)
    east, south = dst_t * (dst_w, dst_h)
    latlon = transform_bounds(DST_CRS, "EPSG:4326", west, south, east, north)

    # Palette PNG: index 0 transparent, index k = class k colour.
    palette = [0, 0, 0] * 256
    for k, v in classes.items():
        palette[k * 3:k * 3 + 3] = hex_rgb(v["color"])

    layers = []
    for l, src in zip(cfg["layers"], srcs):
        a = src.read(1)
        nod = src.nodata
        valid = np.isin(a, list(classes)) if nod is None else (a != nod)
        a = np.where(valid, a, NODATA).astype(np.uint8)

        # Area stats from the native (equal-area-ish UTM) grid.
        px_ha = abs(src.res[0] * src.res[1]) / 10_000
        counts = {str(k): round(float((a == k).sum()) * px_ha, 1) for k in classes}

        dst = np.zeros((dst_h, dst_w), np.uint8)
        reproject(a, dst, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=NODATA, dst_transform=dst_t, dst_crs=DST_CRS,
                  dst_nodata=NODATA, resampling=Resampling.nearest)

        img = Image.fromarray(dst, mode="P")
        img.putpalette(palette)
        name = f"{l['year']}.png"
        img.save(OUT / name, optimize=True, transparency=0)
        layers.append({"year": l["year"], "type": l["type"], "src": f"data/{name}",
                       "area_ha": counts})
        print(f"{l['file']} -> {name}  ({(OUT / name).stat().st_size / 1e6:.2f} MB)")
        src.close()

    meta = {
        "bounds": [[latlon[1], latlon[0]], [latlon[3], latlon[2]]],  # [[S,W],[N,E]]
        "classes": {str(k): v for k, v in classes.items()},
        "layers": layers,
    }
    (OUT / "lulc.js").write_text("window.LULC = " + json.dumps(meta, indent=1) + ";\n")
    print("wrote", OUT / "lulc.js")


if __name__ == "__main__":
    main()
