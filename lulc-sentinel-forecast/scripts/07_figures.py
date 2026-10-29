"""Step 7: every figure for a site, from the committed tables and cached rasters.

City and region sites share the forecast figures; the classification figures
are city-only and the product-comparison figures region-only.

Usage:
    python scripts/07_figures.py --site ljubljana
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lulckit.grid import aoi, geobox, read_tif  # noqa: E402
from lulckit.plot import (GRID, INK, INK2, SERIES, class_legend, plt, save,  # noqa: E402
                          scalebar, show_map)
from lulckit.sites import CLASS_COLORS, CLASSES, site  # noqa: E402

FS_ORDER = ["S2-4b-1s", "S2-10b-1s", "S2-10b+idx-1s", "S2-multiseason",
            "S2+S1-multiseason", "S1-multiseason", "L8-1s", "L8-multiseason"]
FS_LABEL = {
    "S2-4b-1s": "S2 4 bands (10 m), 1 season",
    "S2-10b-1s": "S2 10 bands, 1 season",
    "S2-10b+idx-1s": "S2 10 bands + indices, 1 season",
    "S2-multiseason": "S2 + indices, 3 seasons",
    "S2+S1-multiseason": "S2 + S1, 3 seasons",
    "S1-multiseason": "S1 only, 3 seasons",
    "L8-1s": "Landsat 8/9 + indices, 1 season",
    "L8-multiseason": "Landsat 8/9 + indices, 3 seasons",
}
MODEL_ORDER = ["CART", "RF", "XGBoost", "LightGBM", "SVM-RBF", "MLP"]


# ---------------------------------------------------------------- city only
def fig_study_area(s):
    gb = geobox(s, 10, pad=200)
    sea = s["single_season"]
    a, _, names = read_tif(s["cache"] / f"s2_{s['ref_year']}_{sea}.tif")
    rgb = np.dstack([a[names.index(b)] for b in ("B4", "B3", "B2")]).astype(float)
    lo, hi = np.nanpercentile(rgb[rgb > 0], [2, 98])
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
    df = pd.read_csv(s["data"] / "sample_points.csv")
    b = gb.boundingbox
    ext = (b.left, b.right, b.bottom, b.top)
    fig, ax = plt.subplots(figsize=(7, 7 * gb.shape[0] / gb.shape[1]))
    ax.imshow(rgb, extent=ext)
    aoi(s).boundary.plot(ax=ax, color="white", lw=1.2)
    for k in sorted(df.y.unique()):
        d = df[df.y == k]
        ax.scatter(d.x_m, d.y_m, s=2.5, color=CLASS_COLORS[k], lw=0,
                   label=f"{CLASSES[k]} (n={len(d)})")
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    ax.set_xlabel(""), ax.set_ylabel("")
    ax.set_title(f"Sentinel-2 {s['ref_year']} {sea} composite and stable-consensus sample",
                 loc="left", color=INK)
    leg = ax.legend(loc="lower right", markerscale=4, fontsize=8, frameon=True)
    leg.get_frame().set_alpha(0.85)
    save(fig, s["figures"] / "fig01_study_area")


def fig_heatmap(s, res):
    p = res.pivot(index="feature_set", columns="model", values="OA")
    p = p.reindex([f for f in FS_ORDER if f in p.index])[[m for m in MODEL_ORDER if m in p.columns]]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(p) + 1.2))
    im = ax.imshow(p.values, cmap="Blues", vmin=np.floor(p.values.min() * 20) / 20, vmax=1,
                   aspect="auto")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > np.nanmean(p.values) + 0.02 else INK)
    ax.set_xticks(range(p.shape[1]), p.columns)
    ax.set_yticks(range(p.shape[0]), [FS_LABEL[f] for f in p.index])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Hold-out overall accuracy")
    ax.set_title("Overall accuracy on spatially held-out blocks", loc="left")
    save(fig, s["figures"] / "fig02_accuracy_heatmap")


def fig_ladder(s, res, best_model):
    d = res[res.model == best_model].set_index("feature_set")
    d = d.reindex([f for f in FS_ORDER if f in d.index])
    fig, ax = plt.subplots(figsize=(6.8, 0.4 * len(d) + 1.0))
    y = np.arange(len(d))[::-1]
    for yi, (fs, r) in zip(y, d.iterrows()):
        col = SERIES[1] if fs.startswith("L8") else SERIES[0]
        ax.plot([r.OA_lo, r.OA_hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(r.OA, yi, "o", ms=7, color=col, mec="white", mew=1.5)
        ax.text(r.OA_hi + 0.004, yi, f"{r.OA:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [FS_LABEL[f] for f in d.index])
    ax.set_xlabel(f"Overall accuracy, {best_model} (95 % block-bootstrap interval)")
    ax.grid(axis="y", visible=False)
    ax.set_title("What each Sentinel input adds (blue) against Landsat 8/9 (orange)", loc="left")
    save(fig, s["figures"] / "fig03_input_ladder")


def fig_perclass(s, best_model):
    pr = pd.read_csv(s["data"] / "test_predictions.csv")
    y = pr["y"].to_numpy()
    fss = [f for f in FS_ORDER if f"{f}|{best_model}" in pr.columns]
    classes = sorted(np.unique(y))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(fss))
    for k in classes:
        f1 = []
        for fs in fss:
            p = pr[f"{fs}|{best_model}"].to_numpy()
            tp = np.sum((y == k) & (p == k))
            ua, pa = tp / max(1, np.sum(p == k)), tp / max(1, np.sum(y == k))
            f1.append(2 * ua * pa / max(1e-9, ua + pa))
        ax.plot(x, f1, "-o", color=CLASS_COLORS[k], lw=2, ms=5, mec="white", mew=1,
                label=CLASSES[k])
    ax.set_xticks(x, [FS_LABEL[f].replace(", ", ",\n") for f in fss], fontsize=7)
    ax.set_ylabel(f"Class F1 ({best_model})")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    ax.set_title("Per-class F1 across feature sets", loc="left")
    save(fig, s["figures"] / "fig04_perclass_f1")


def fig_transfer(s):
    t = pd.read_csv(s["tables"] / "temporal_transfer.csv")
    t = t[t.feature_set.isin(["S2-10b+idx-1s", "S2-multiseason", "S2+S1-multiseason"])]
    yrs = sorted(t.test_year.unique())
    fig, axes = plt.subplots(1, len(yrs), figsize=(7.5, 2.8), sharey=True)
    for ax, yr in zip(axes, yrs):
        d = t[t.test_year == yr]
        fss = list(dict.fromkeys(d.feature_set))
        x = np.arange(len(fss))
        tr = d[d.train_year == s["ref_year"]].set_index("feature_set").reindex(fss)
        ax.bar(x - 0.18, tr.OA, 0.34, color=SERIES[0], label=f"trained on {s['ref_year']}")
        own = d[d.train_year == yr].set_index("feature_set").reindex(fss)
        if yr != s["ref_year"]:
            ax.bar(x + 0.18, own.OA, 0.34, color=SERIES[2], label="retrained on own year")
        ax.set_xticks(x, ["1 season", "3 seasons", "3 seasons\n+ S1"], fontsize=7)
        ax.set_title(f"Test year {yr}", loc="left")
        ax.set_ylim(max(0, np.nanmin(d.OA) - 0.08), 1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Overall accuracy")
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    save(fig, s["figures"] / "fig05_temporal_transfer")


def fig_maps(s, cfg):
    yrs = s["years"]
    m0 = read_tif(s["cache"] / f"map10_{yrs[0]}_{cfg['model']}.tif")[0]
    rows, cols = np.nonzero(m0)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    asp = (r1 - r0) / (c1 - c0)
    fig, axes = plt.subplots(1, len(yrs), figsize=(11, 11 / len(yrs) * asp + 0.9))
    for ax, y in zip(axes, yrs):
        m = read_tif(s["cache"] / f"map10_{y}_{cfg['model']}.tif")[0][r0:r1, c0:c1]
        show_map(ax, m[::2, ::2], title=str(y))
    scalebar(axes[0], 20, 5)
    fig.subplots_adjust(wspace=0.02)
    class_legend(fig, cfg["classes"], loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{s['title']}: Sentinel-2 maps ({cfg['feature_set']}, {cfg['model']})",
                 x=0.01, ha="left", color=INK)
    save(fig, s["figures"] / "fig06_maps")


# ---------------------------------------------------------------- forecast (all)
def fig_validation(s):
    v = pd.read_csv(s["tables"] / "forecast_validation.csv")
    v = v[v.variant != "Null (persistence)"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw=dict(width_ratios=[1, 1.3]))
    ax = axes[0]
    y = np.arange(len(v))[::-1]
    ax.barh(y + 0.18, v.FoM, 0.34, color=SERIES[0], label="all transitions")
    ax.barh(y - 0.18, v.FoM_built, 0.34, color=SERIES[1], label="gain of built-up")
    ax.set_yticks(y, v.variant)
    ax.set_xlabel("Figure of merit (out-of-time)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Hindcast accuracy of change", loc="left")
    ax = axes[1]
    comp = ["hits", "wrong_hits", "misses", "false_alarms"]
    cols = [SERIES[2], SERIES[3], SERIES[1], SERIES[4]]
    tot = v[comp].sum(1).to_numpy()
    left = np.zeros(len(v))
    for c, col in zip(comp, cols):
        w = v[c].to_numpy() / tot
        ax.barh(y, w, 0.6, left=left, color=col, label=c.replace("_", " "),
                edgecolor="white", lw=1)
        left += w
    ax.set_yticks(y, [""] * len(y))
    ax.set_xlabel("Share of observed or simulated change")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="y", visible=False)
    ax.set_title("Components (Pontius)", loc="left")
    save(fig, s["figures"] / "fig07_forecast_validation")


def fig_importance(s):
    d = pd.read_csv(s["tables"] / "driver_importance.csv")
    d = d[d.variant == "RF-CA"]
    if d.empty:
        return
    p = d.pivot(index="driver", columns="name", values="importance").clip(lower=0)
    p = p.loc[p.max(1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.0 * p.shape[1] + 2.5, 0.38 * p.shape[0] + 1))
    im = ax.imshow(p.values, cmap="Blues", aspect="auto", vmin=0)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > np.nanmax(p.values) * 0.6 else INK)
    ax.set_xticks(range(p.shape[1]), [f"to {c}" for c in p.columns], rotation=30, ha="right")
    ax.set_yticks(range(p.shape[0]), [r.replace("_", " ") for r in p.index])
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04).set_label("Permutation importance (AUC drop)")
    ax.set_title("Which drivers locate each transition (RF potential)", loc="left")
    save(fig, s["figures"] / "fig08_driver_importance")


def fig_forecast_maps(s):
    cfg = json.load(open(s["tables"] / "forecast_config.json"))
    t2, hz = cfg["epochs"][2], cfg["horizons"]
    if s["kind"] == "city":
        m0 = read_tif(s["cache"] / f"map30_{t2}.tif")[0]
    else:
        src = "clc" if s["source"] == "corine" else "cci"
        m0 = read_tif(s["cache"] / f"map_{src}_{t2}.tif")[0]
    panels = [(m0, f"Observed {t2}")]
    panels += [(read_tif(s["out"] / f"forecast_bau_{h}.tif")[0], f"BAU {h}") for h in hz[::2]]
    panels += [(read_tif(s["out"] / f"forecast_ecological_{hz[-1]}.tif")[0],
                f"Ecological protection {hz[-1]}"),
               (read_tif(s["out"] / f"forecast_compact_{hz[-1]}.tif")[0], f"Compact growth {hz[-1]}")]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.2))
    for ax, (m, t) in zip(axes, panels):
        show_map(ax, m, title=t)
    class_legend(fig, cfg["classes"], loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06))
    save(fig, s["figures"] / "fig09_forecast_maps")


def fig_trajectories(s):
    fa = pd.read_csv(s["tables"] / "forecast_areas.csv")
    cfg = json.load(open(s["tables"] / "forecast_config.json"))
    t2 = cfg["epochs"][2]
    show = [2, 3, 4] if s["kind"] == "region" else [2, 3, 4, 5]
    show = [k for k in show if k in cfg["classes"]]
    fig, axes = plt.subplots(1, len(show), figsize=(2.6 * len(show), 2.8))
    sc_col = {"BAU": SERIES[0], "Ecological protection": SERIES[2], "Compact growth": SERIES[1]}
    for ax, k in zip(axes, show):
        ob = fa[(fa.scenario == "observed") & (fa.class_id == k)].sort_values("year")
        last = float(ob[ob.year == t2].area_km2.iloc[0])
        ax.plot(ob.year, ob.area_km2, "-o", color=INK, lw=2, ms=5, label="observed")
        for sc, col in sc_col.items():
            d = fa[(fa.scenario == sc) & (fa.markov == "t1t2") & (fa.class_id == k)].sort_values("year")
            ax.plot([t2] + d.year.tolist(), [last] + d.area_km2.tolist(), "-", color=col, lw=2,
                    label=sc)
        alt = fa[(fa.scenario == "BAU") & (fa.markov == "t0t2") & (fa.class_id == k)].sort_values("year")
        bau = fa[(fa.scenario == "BAU") & (fa.markov == "t1t2") & (fa.class_id == k)].sort_values("year")
        ax.fill_between([t2] + bau.year.tolist(),
                        [last] + np.minimum(bau.area_km2, alt.area_km2.values).tolist(),
                        [last] + np.maximum(bau.area_km2, alt.area_km2.values).tolist(),
                        color=SERIES[0], alpha=0.15, lw=0)
        ax.set_title(CLASSES[k], loc="left")
        ax.set_ylabel("km²" if ax is axes[0] else "")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Area trajectories; shaded band = BAU under the alternative Markov matrix",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    save(fig, s["figures"] / "fig10_trajectories")


# ---------------------------------------------------------------- region only
def fig_region_history(s):
    t = pd.read_csv(s["tables"] / "class_areas_history.csv")
    src_main = "CLC" if s["source"] == "corine" else "CCI"
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9))
    for ax, k in zip(axes, [2, 3, 4]):
        for src, col, mk in ((src_main, INK, "o"), ("IO", SERIES[0], "s"),
                             ("WorldCover", SERIES[1], "D")):
            d = t[(t.source == src) & (t.class_id == k)].sort_values("year")
            if len(d):
                ax.plot(d.year, d.share * 100, "-" + mk, color=col, lw=2, ms=5, mec="white",
                        label={"CLC": "CORINE (100 m)", "CCI": "ESA CCI (300 m)",
                               "IO": "Impact Observatory (S2, 10 m)",
                               "WorldCover": "ESA WorldCover (S2+S1, 10 m)"}[src])
        ax.set_title(CLASSES[k], loc="left")
        ax.set_ylabel("% of area" if ax is axes[0] else "")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{s['title']}: long-record product vs Sentinel-era products",
                 x=0.01, ha="left", color=INK)
    save(fig, s["figures"] / "fig02_products")


def fig_region_epochs(s):
    src = "clc" if s["source"] == "corine" else "cci"
    cfg = json.load(open(s["tables"] / "forecast_config.json"))
    yrs = cfg["epochs"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    for ax, y in zip(axes, yrs):
        show_map(ax, read_tif(s["cache"] / f"map_{src}_{y}.tif")[0], title=f"{src.upper()} {y}")
    show_map(axes[3], read_tif(s["cache"] / "map_io_2023.tif")[0], title="IO (Sentinel-2) 2023")
    class_legend(fig, cfg["classes"], loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05))
    save(fig, s["figures"] / "fig01_epochs")


def fig_units(s):
    import geopandas as gpd
    u = gpd.read_file(s["cache"] / "units.gpkg")
    col = u.columns[0]
    t = pd.read_csv(s["tables"] / "units_summary.csv")
    u = u.merge(t, left_on=col, right_on="unit")
    u["hist"] = 100 * u["built_growth_hist_km2"] / u["area_km2"]
    u["fcst"] = 100 * u["built_growth_fcst_km2"] / u["area_km2"]
    vmax = np.nanpercentile(np.r_[u.hist, u.fcst], 98)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    cfg = json.load(open(s["tables"] / "forecast_config.json"))
    t0, t2, h = cfg["epochs"][0], cfg["epochs"][2], cfg["horizons"][-1]
    for ax, c, ttl in ((axes[0], "hist", f"Observed built-up gain {t0}-{t2}"),
                       (axes[1], "fcst", f"Forecast built-up gain {t2}-{h} (BAU)")):
        u.plot(column=c, ax=ax, cmap="Reds", vmin=0, vmax=vmax, edgecolor="white", lw=0.3,
               legend=True, legend_kwds=dict(label="% of unit area", shrink=0.6))
        ax.set_axis_off()
        ax.set_title(ttl, loc="left")
    save(fig, s["figures"] / "fig11_units")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    s = site(a.site)
    jobs = []
    if s["kind"] == "city":
        res = pd.read_csv(s["tables"] / "classifier_featureset.csv")
        cfg = json.load(open(s["tables"] / "best_config.json"))
        jobs += [("study", lambda: fig_study_area(s)), ("heatmap", lambda: fig_heatmap(s, res)),
                 ("ladder", lambda: fig_ladder(s, res, cfg["model"])),
                 ("perclass", lambda: fig_perclass(s, cfg["model"])),
                 ("transfer", lambda: fig_transfer(s)), ("maps", lambda: fig_maps(s, cfg))]
    else:
        jobs += [("products", lambda: fig_region_history(s)),
                 ("epochs", lambda: fig_region_epochs(s)), ("units", lambda: fig_units(s))]
    jobs += [("validation", lambda: fig_validation(s)), ("importance", lambda: fig_importance(s)),
             ("forecast", lambda: fig_forecast_maps(s)), ("traj", lambda: fig_trajectories(s))]
    for name, fn in jobs:
        if a.only and name not in a.only:
            continue
        try:
            fn()
            print(f"  {name}")
        except FileNotFoundError as e:
            print(f"  {name}: skipped ({e.filename})")


if __name__ == "__main__":
    main()
