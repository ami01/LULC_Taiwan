# Sentinel-era land cover and land-change forecasting: Varanasi, Uttar Pradesh, Ljubljana, Slovenia

Code, tables, figures and manuscript drafts for four linked studies that extend
the Taipei classifier comparison (`ami01/taipei-lulc`) and the Taiwan Markov
forecast (`ami01/LULC_Taiwan`, the `*_LULC.tif` files one level up):

| Study | Scale | Question | Manuscript |
|---|---|---|---|
| **Varanasi district** (1,545 km²) | city, 10 m | How much do Sentinel-2/-1 inputs and the choice of classifier matter, and where is the city growing? | short paper |
| **Uttar Pradesh** (241,000 km²) | state, 300 m | What do long-record and Sentinel-era products say about land change since 1995, and what follows to 2050 under three scenarios? | full paper |
| **Ljubljana** (275 km²) | city, 10 m | Same as Varanasi, plus an independent check against LUCAS in-situ points | short paper |
| **Slovenia** (20,273 km²) | country, 100 m | Same as Uttar Pradesh, on CORINE 1990-2018, with LUCAS validation | full paper |

Drafts are in `manuscripts/<site>/paper.tex`, each with its own `refs.bib`
(every entry fetched from Crossref/DataCite with `crossref-bib`) and
`claims.jsonl` (the claims ledger from `ami01/manuscript-kit`).

## Design in one paragraph

Cities: seasonal Sentinel-2 L2A, Sentinel-1 RTC and Landsat 8/9 composites
are built from STAC catalogues; labels come from pixels where ESA WorldCover
2021 and the Impact Observatory maps for 2018, 2021 and 2023 all agree (no
field data exist for these sites); six classifiers are tuned on eight feature
sets and compared on spatially held-out blocks with block-bootstrap intervals
and cluster-corrected McNemar tests; the best configuration maps 2018, 2021 and
2024. Forecasts, at every site: a Markov chain sets the quantity of change, a
transition-potential model (logistic regression, ANN or random forest) on eight
drivers says where, and a demand-constrained cellular automaton allocates it.
Every variant is validated out of time (calibrate t0->t1, predict t2) with the
figure of merit against null models, before forecasting three scenarios.

## Reproducing

Python 3.11+. No Earth Engine account is needed; all imagery is read from
public STAC catalogues (Earth Search on AWS, Microsoft Planetary Computer).

```bash
pip install -r env/requirements.txt

# city sites (varanasi, ljubljana)
python scripts/01_composites.py --site varanasi      # ~1.5 h of downloads
python scripts/02_labels.py     --site varanasi
python scripts/03_classify.py   --site varanasi      # ~1 h on 4 cores
python scripts/04_map.py        --site varanasi
python scripts/06_forecast.py   --site varanasi
python scripts/07_figures.py    --site varanasi

# region sites (uttar_pradesh, slovenia)
python scripts/05_regional_maps.py --site slovenia
python scripts/06_forecast.py      --site slovenia
python scripts/07_figures.py       --site slovenia

# LUCAS in-situ check (slovenia, ljubljana)
python scripts/08_lucas.py --site ljubljana
```

Everything a paper reports is in `results/<site>/tables/` and regenerates from
the scripts. The sampled feature tables (`data/<site>/features_<year>.csv`)
are committed, so step 3 reruns offline. Large rasters (composites, maps,
drivers) go to `data/cache/`, which is not committed; forecast maps are in
`results/<site>/`.

`lulckit/sites.py` holds every site setting: boundary, CRS, epochs, seasons,
the harmonised six-class scheme and the cross-walks from WorldCover, Impact
Observatory, ESA CCI and CORINE.

## Things worth knowing before you trust a number

* **Labels are map consensus, not ground truth.** Hold-out accuracy on stable
  consensus pixels is an upper bound on map accuracy. For Ljubljana and
  Slovenia, `08_lucas.py` gives agreement with in-situ LUCAS points; for
  Varanasi and Uttar Pradesh no such data exist and the papers say so.
* **Sentinel-2 offset.** Earth Search marks post-2022 scenes
  `boa_offset_applied = True` *and* declares `offset = -0.1` on each band. The
  offset is already removed from the pixels; applying it again zeroes the
  visible bands (see the docstring of `lulckit/composite.py`).
* **Built-up area in the Gangetic plain is contested between products.** For
  Uttar Pradesh in 2020-2021, ESA CCI gives 1.1 %, WorldCover 2.5 % and Impact
  Observatory 9.4 % built-up. The forecast inherits whichever product it is
  fitted to; the UP paper treats this as a result, not a footnote.
* **Drivers are present-day.** Roads, railways and settlements are 2024-2026
  snapshots used for every period, which flatters the hindcasts slightly.

## Data sources

| Data | Source | Licence |
|---|---|---|
| Sentinel-2 L2A | Earth Search `sentinel-2-l2a` (AWS) | Copernicus open |
| Sentinel-1 RTC | Planetary Computer `sentinel-1-rtc` | Copernicus open / CC BY 4.0 |
| Landsat 8/9 C2 L2 | Planetary Computer `landsat-c2-l2` | public domain |
| ESA WorldCover 2021 v200 | Planetary Computer `esa-worldcover` | CC BY 4.0 |
| Impact Observatory annual LULC v02 | Planetary Computer `io-lulc-annual-v02` | CC BY 4.0 |
| ESA CCI Land Cover | Planetary Computer `esa-cci-lc` | ESA CCI licence |
| CORINE Land Cover 1990-2018 | EEA discomap vector services | Copernicus open |
| Copernicus GLO-30 DEM | Planetary Computer `cop-dem-glo-30` | Copernicus DEM licence |
| JRC Global Surface Water | Planetary Computer `jrc-gsw` | Copernicus open |
| WorldPop 1 km, 2000-2020 | data.worldpop.org | CC BY 4.0 |
| OpenStreetMap | openstreetmap.fr / BBBike extracts | ODbL |
| GRIP4 roads (Slovenia) | PBL Netherlands | CC0 |
| GeoNames (Slovenia settlements) | geonames.org | CC BY 4.0 |
| LUCAS Copernicus 2018, 2022 | JRC open data | Eurostat reuse policy |
| Boundaries | geoBoundaries gbOpen | CC BY 4.0 |
