"""Step 6 (all sites): calibrate, validate and run the land-change forecast.

Three epochs (t0, t1, t2) per site:
    Varanasi, Ljubljana   2018, 2021, 2024   (Sentinel-2 maps from step 4, 30 m)
    Uttar Pradesh         2000, 2010, 2020   (ESA CCI, 300 m)
    Slovenia              2006, 2012, 2018   (CORINE, 100 m)

1. Calibrate on t0 -> t1: Markov matrix, transition-potential models, and the
   allocation parameters (neighbourhood weight w, conversion resistance e) by
   in-sample figure of merit.
2. Validate out of time on t1 -> t2: start from the observed t1 map, project
   with what was learned on t0 -> t1, compare with the observed t2 map.
   Eight variants are compared:
       Null (persistence), Markov + random allocation, CA (neighbourhood only),
       LR-CA, ANN-CA, RF-CA, and RF-CA with the three-variable and the static
       (no dynamic distance) driver sets.
3. Forecast from t2 with everything re-learned on t1 -> t2, under three
   scenarios (business as usual, ecological protection, infill-only growth),
   with a quantity band from the alternative Markov matrix (t0 -> t2).

Usage:
    python scripts/06_forecast.py --site varanasi
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lulckit import drivers as D  # noqa: E402
from lulckit import forecast as FC  # noqa: E402
from lulckit.grid import read_tif, write_tif  # noqa: E402
from lulckit.sites import CLASSES, RAW, site  # noqa: E402

EPOCHS = {"varanasi": (2018, 2021, 2024), "ljubljana": (2018, 2021, 2024),
          "uttar_pradesh": (2000, 2010, 2020), "slovenia": (2006, 2012, 2018)}
HORIZONS = {"varanasi": (2030, 2035, 2040), "ljubljana": (2030, 2035, 2040),
            "uttar_pradesh": (2030, 2040, 2050), "slovenia": (2030, 2040, 2050)}
STEP = {"varanasi": 1, "ljubljana": 1, "uttar_pradesh": 5, "slovenia": 3}
W_GRID = (0.0, 0.4, 0.8)
E_GRID = (0.2, 0.5, 0.8, 0.95)


def load_maps(s, years):
    out = {}
    for y in years:
        if s["kind"] == "city":
            p = s["cache"] / f"map30_{y}.tif"
        else:
            p = s["cache"] / f"map_{'clc' if s['source'] == 'corine' else 'cci'}_{y}.tif"
        m, gb, _ = read_tif(p)
        out[y] = m
    return out, gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    a = ap.parse_args()
    s = site(a.site)
    t0, t1, t2 = EPOCHS[s["name"]]
    step = STEP[s["name"]]
    maps, gb = load_maps(s, (t0, t1, t2))
    res = int(abs(gb.affine.a))
    mask = (maps[t0] > 0) & (maps[t1] > 0) & (maps[t2] > 0)
    classes = [k for k in CLASSES if any(np.any(maps[y][mask] == k) for y in maps)]
    print(f"{s['name']}: grid {gb.shape} at {res} m, {mask.sum()} cells, classes {classes}")

    BUILT = 2
    allowed = {(i, j): not (i == BUILT and j != BUILT) for i in classes for j in classes}
    drv = {y: D.build(s, gb, y) for y in (t0, t1, t2)}
    names_full = [n for n in D.STATIC if n in drv[t0]]
    tabs = s["tables"]

    # ---------------- Markov ----------------
    C01 = FC.transition_counts(maps[t0], maps[t1], classes, mask)
    C12 = FC.transition_counts(maps[t1], maps[t2], classes, mask)
    C02 = FC.transition_counts(maps[t0], maps[t2], classes, mask)
    P01 = FC.annualize(FC.transition_matrix(C01), t1 - t0)
    P12 = FC.annualize(FC.transition_matrix(C12), t2 - t1)
    P02 = FC.annualize(FC.transition_matrix(C02), t2 - t0)
    lab = [CLASSES[k] for k in classes]
    cell_km2 = (res / 1000) ** 2
    for nm, C in (("t0t1", C01), ("t1t2", C12)):
        pd.DataFrame(C * cell_km2, index=lab, columns=lab).to_csv(
            tabs / f"transition_km2_{nm}.csv", float_format="%.3f")
    for nm, P in (("t0t1", P01), ("t1t2", P12), ("t0t2", P02)):
        pd.DataFrame(P, index=lab, columns=lab).to_csv(
            tabs / f"markov_annual_{nm}.csv", float_format="%.6f")

    # ---------------- potentials, calibration period ----------------
    variants = {
        "LR-CA": ("LR", names_full, True),
        "ANN-CA": ("MLP", names_full, True),
        "RF-CA": ("RF", names_full, True),
        "RF-CA (3 var)": ("RF", [n for n in D.THREE_VAR if n in names_full], False),
        "RF-CA (static)": ("RF", names_full, False),
    }
    pots, auc_rows, imp_rows = {}, [], []
    for v, (kind, names, dyn) in variants.items():
        pot = FC.Potential(kind, classes, res, names, dynamic=dyn).fit(
            maps[t0], maps[t1], drv[t0], mask, allowed)
        out = pot.auc_out(maps[t1], maps[t2], drv[t1], mask, allowed)
        for k in classes:
            if k in pot.auc_in:
                auc_rows.append(dict(variant=v, class_id=k, name=CLASSES[k],
                                     auc_calibration=pot.auc_in[k],
                                     auc_validation=out.get(k, np.nan)))
        for k, d in pot.importance.items():
            for f, val in d.items():
                imp_rows.append(dict(variant=v, class_id=k, name=CLASSES[k],
                                     driver=f, importance=val))
        pots[v] = pot
        print(f"  potentials {v}: AUC in {np.round(list(pot.auc_in.values()), 3)}", flush=True)
    pd.DataFrame(auc_rows).to_csv(tabs / "potential_auc.csv", index=False, float_format="%.4f")
    pd.DataFrame(imp_rows).to_csv(tabs / "driver_importance.csv", index=False, float_format="%.5f")

    def elas(e):
        return {k: (1.0 if k == BUILT else e) for k in classes}

    # ---------------- calibrate w, e in sample (t0 -> t1) ----------------
    G0 = {v: p.surfaces(maps[t0], drv[t0], mask) for v, p in pots.items()}
    G0["CA"] = np.ones_like(G0["RF-CA"])
    cal, best = [], {}
    for v, G in G0.items():
        for w, e in itertools.product(W_GRID, E_GRID):
            if v == "CA" and w == 0.0:
                continue
            sim, _ = FC.simulate(maps[t0], P01, t1 - t0, G, classes, mask, elas(e),
                                 allowed, w=w, step=step)
            r = FC.pontius(maps[t0], maps[t1], sim, mask, classes)
            cal.append(dict(variant=v, w=w, e=e, **r))
        c = pd.DataFrame([x for x in cal if x["variant"] == v])
        b = c.sort_values("FoM", ascending=False).iloc[0]
        best[v] = (float(b.w), float(b.e))
        print(f"  calibrated {v}: w={b.w} e={b.e} FoM_in={b.FoM:.4f}", flush=True)
    pd.DataFrame(cal).to_csv(tabs / "calibration_grid.csv", index=False, float_format="%.5f")

    # ---------------- validate out of time (t1 -> t2) ----------------
    val = []
    a_t1 = FC.class_areas(maps[t1], classes, mask)
    dem = FC.project(a_t1, P01, t2 - t1)
    sims = {"Null (persistence)": maps[t1].copy(),
            "Markov + random": FC.random_allocation(maps[t1], dem, classes, mask, allowed)}
    G1 = {v: p.surfaces(maps[t1], drv[t1], mask) for v, p in pots.items()}
    G1["CA"] = np.ones_like(G1["RF-CA"])
    for v, G in G1.items():
        w, e = best[v]
        sims[v], _ = FC.simulate(maps[t1], P01, t2 - t1, G, classes, mask, elas(e),
                                 allowed, w=w, step=step)
    for v, sim in sims.items():
        r = FC.pontius(maps[t1], maps[t2], sim, mask, classes)
        rb = FC.pontius(maps[t1], maps[t2], sim, mask, focus=BUILT)
        val.append(dict(variant=v, w=best.get(v, (np.nan,))[0],
                        e=best.get(v, (np.nan, np.nan))[1], **r,
                        FoM_built=rb["FoM"], built_hits=rb["hits"],
                        built_misses=rb["misses"], built_false_alarms=rb["false_alarms"]))
        if v == "RF-CA":
            write_tif(s["out"] / f"validation_sim_{t2}.tif", sim.astype("uint8"), gb, nodata=0)
    vt = pd.DataFrame(val)
    vt.to_csv(tabs / "forecast_validation.csv", index=False, float_format="%.5f")
    print(vt[["variant", "FoM", "FoM_built", "agreement", "quantity_dis",
              "allocation_dis", "null_agreement"]].round(4).to_string(index=False))
    mr = []
    for v in ("Null (persistence)", "Markov + random", "LR-CA", "ANN-CA", "RF-CA"):
        d = FC.multires_agreement(maps[t2], sims[v], mask, classes)
        d.insert(0, "variant", v)
        mr.append(d)
    pd.concat(mr).to_csv(tabs / "forecast_multires.csv", index=False, float_format="%.5f")

    # ---------------- forecast from t2 ----------------
    ml = vt[~vt.variant.isin(["Null (persistence)", "Markov + random", "CA"])]
    chosen = ml.sort_values("FoM", ascending=False).iloc[0]["variant"]
    kind, names, dyn = variants[chosen]
    potF = FC.Potential(kind, classes, res, names, dynamic=dyn).fit(
        maps[t1], maps[t2], drv[t1], mask, allowed)
    GF = potF.surfaces(maps[t2], drv[t2], mask)
    w, e = best[chosen]
    dist_built = FC.dist_to_class(maps[t2], BUILT, res)
    # Scenario rules are resolution-aware so that they bind: at 300 m nearly
    # all growth is already within 1 km of built-up land, so a distance rule
    # of that size would leave the forecast unchanged.
    #   Infill: new built-up only where >= 50 % of the 5 x 5 neighbourhood is
    #   already built (adjacency alone does not bind: BAU growth is already
    #   contiguous with the existing edge at every site).
    #   Ecological protection: tree cover, water and grass/shrub frozen, and no
    #   new built-up within a riparian buffer (100 m, or one cell if coarser).
    riparian = max(100.0, float(res))
    scen = {
        "BAU": dict(),
        "Ecological protection": dict(
            restrict=np.isin(maps[t2], [1, 4, 5]),
            forbid=lambda m: {BUILT: FC.dist_to_class(m, 1, res) <= riparian}),
        "Infill": dict(forbid=lambda m: {
            BUILT: FC.neighbourhood(m, [BUILT])[0] < 0.5}),
    }
    hz = HORIZONS[s["name"]]
    rows = []
    a_t2 = FC.class_areas(maps[t2], classes, mask)
    for sc, kw in scen.items():
        for Pnm, P in (("t1t2", P12), ("t0t2", P02)):
            if Pnm == "t0t2" and sc != "BAU":
                continue
            _, keep = FC.simulate(maps[t2], P, hz[-1] - t2, GF, classes, mask, elas(e),
                                  allowed, w=w, return_steps=[h - t2 for h in hz],
                                  step=step, **kw)
            for h in hz:
                m = keep[h - t2]
                for i, k in enumerate(classes):
                    rows.append(dict(scenario=sc, markov=Pnm, year=h, class_id=k,
                                     name=CLASSES[k],
                                     area_km2=float(np.sum(m[mask] == k) * cell_km2),
                                     markov_km2=float(FC.project(a_t2, P, h - t2)[i] * cell_km2)))
                if Pnm == "t1t2":
                    tag = sc.split()[0].lower()
                    write_tif(s["out"] / f"forecast_{tag}_{h}.tif", m.astype("uint8"), gb, nodata=0)
    for i, k in enumerate(classes):
        for y in (t0, t1, t2):
            rows.append(dict(scenario="observed", markov="", year=y, class_id=k, name=CLASSES[k],
                             area_km2=float(np.sum(maps[y][mask] == k) * cell_km2),
                             markov_km2=np.nan))
    fa = pd.DataFrame(rows)
    fa.to_csv(tabs / "forecast_areas.csv", index=False, float_format="%.4f")
    print(fa[(fa.markov != "t0t2")].pivot_table(index=["scenario", "year"], columns="name",
                                                 values="area_km2").round(1))

    # District / municipality summary for regional sites.
    if s["kind"] == "region":
        import geopandas as gpd
        from lulckit.grid import aoi
        fn, col = s["districts"]
        dist = gpd.read_file(RAW / "boundaries" / fn).to_crs(s["crs"])
        reg = aoi(s).geometry.iloc[0]
        dist = dist[dist.representative_point().within(reg)].reset_index(drop=True)
        zid = rasterize([(g, i + 1) for i, g in enumerate(dist.geometry)], out_shape=gb.shape,
                        transform=gb.affine, fill=0, dtype="int32")
        fut = read_tif(s["out"] / f"forecast_bau_{hz[-1]}.tif")[0]
        recs = []
        for i, name in enumerate(dist[col]):
            z = (zid == i + 1) & mask
            if z.sum() == 0:
                continue
            r = dict(unit=name, area_km2=float(z.sum() * cell_km2))
            for y, m in ((t0, maps[t0]), (t2, maps[t2]), (hz[-1], fut)):
                r[f"built_{y}_km2"] = float(np.sum(m[z] == BUILT) * cell_km2)
                r[f"crop_{y}_km2"] = float(np.sum(m[z] == 3) * cell_km2)
                r[f"tree_{y}_km2"] = float(np.sum(m[z] == 4) * cell_km2)
            recs.append(r)
        dt = pd.DataFrame(recs)
        dt["built_growth_hist_km2"] = dt[f"built_{t2}_km2"] - dt[f"built_{t0}_km2"]
        dt["built_growth_fcst_km2"] = dt[f"built_{hz[-1]}_km2"] - dt[f"built_{t2}_km2"]
        dt["crop_change_fcst_km2"] = dt[f"crop_{hz[-1]}_km2"] - dt[f"crop_{t2}_km2"]
        dt.sort_values("built_growth_fcst_km2", ascending=False).to_csv(
            tabs / "units_summary.csv", index=False, float_format="%.3f")
        dist[[col, "geometry"]].to_file(s["cache"] / "units.gpkg", driver="GPKG")

    json.dump(dict(epochs=[t0, t1, t2], horizons=list(hz), chosen=chosen, w=w, e=e,
                   classes=classes, res=res, drivers=names_full, cells=int(mask.sum())),
              open(tabs / "forecast_config.json", "w"), indent=2)
    print(f"Forecast model: {chosen} (w={w}, e={e})")


if __name__ == "__main__":
    main()
