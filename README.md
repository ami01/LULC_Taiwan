# Taiwan LULC Explorer

Interactive web viewer for historical (1990, 2000, 2010, 2020) and projected
(2030, 2050) land use / land cover maps of Taiwan.

**Live site:** https://ami01.github.io/LULC_Taiwan/

- **Timeline** – slide or play through all years.
- **Compare** – pick any two years and drag the swipe handle.
- Area table and trend chart per class (projected years dashed/shaded).

## Repository layout

| Path | Purpose |
|---|---|
| `*_LULC.tif` | Source rasters (EPSG:32651, 100 m) |
| `tools/classes.json` | Class names, colours, and layer list |
| `tools/convert_lulc.py` | Converts GeoTIFFs → `docs/data/*.png` + `docs/data/lulc.js` |
| `docs/` | The static website served by GitHub Pages |

## Updating class names / colours

1. Edit `tools/classes.json`.
2. Regenerate the web data:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install rasterio numpy pillow
   python tools/convert_lulc.py
   ```
3. Commit and push. Pages redeploys automatically.

## Publishing on GitHub Pages (one-time)

Repo **Settings → Pages → Build and deployment → Source: Deploy from a branch**,
branch `main`, folder **`/docs`** → Save. The site appears at the URL above
within a minute or two. (Free Pages requires the repo to be public.)

Open `docs/index.html` directly in a browser to preview locally.
