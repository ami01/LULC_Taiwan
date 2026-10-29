"""Step 2 (city sites): reference products, stable consensus, training sample.

Writes data/cache/<site>/ref_*.tif and consensus.tif, and
data/<site>/sample_points.csv (row, col, label, spatial block, split).

Usage:
    python scripts/02_labels.py --site varanasi
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lulckit import labels as L  # noqa: E402
from lulckit.grid import aoi_mask, geobox, read_tif, write_tif  # noqa: E402
from lulckit.sites import CLASSES, site  # noqa: E402

N_PER_CLASS = 1500
MIN_CLASS_N = 100
TEST_FOLDS = 3          # of 10 grouped folds -> ~30 % of blocks held out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    a = ap.parse_args()
    s = site(a.site)
    gb = geobox(s, 10, pad=200)
    mask = aoi_mask(s, gb)

    io_years = sorted({min(y, 2023) for y in s["years"]})
    refs = {}
    for coll, yrs in (("esa-worldcover", [2021]), ("io-lulc-annual-v02", io_years)):
        for y in yrs:
            tag = f"{'wc' if coll == 'esa-worldcover' else 'io'}{y}"
            p = s["cache"] / f"ref_{tag}.tif"
            if not p.exists():
                raw = L.load_product(gb, coll, y)
                write_tif(p, L.to_scheme(raw, coll), gb, nodata=0)
                print(f"  {tag} done")
            refs[tag] = read_tif(p)[0]

    cons = L.consensus([refs["wc2021"]] + [refs[f"io{y}"] for y in io_years], mask)
    write_tif(s["cache"] / "consensus.tif", cons, gb, nodata=0)

    # How restrictive is the consensus? Report per class.
    rows = []
    for k, name in CLASSES.items():
        rows.append(dict(class_id=k, name=name,
                         wc2021_share=round(float((refs["wc2021"][mask] == k).mean()), 4),
                         io2021_share=round(float((refs["io2021"][mask] == k).mean()), 4),
                         consensus_pixels=int((cons == k).sum()),
                         consensus_frac_of_wc=round(float((cons == k).sum() /
                                                  max(1, (refs["wc2021"][mask] == k).sum())), 4)))
    tab = pd.DataFrame(rows)
    agree = float((refs["wc2021"][mask] == refs["io2021"][mask]).mean())
    tab.to_csv(s["tables"] / "consensus_frame.csv", index=False)
    print(tab.to_string(index=False))
    print(f"WorldCover 2021 vs IO 2021 per-pixel agreement in AOI: {agree:.3f}")
    print(f"Consensus covers {float((cons > 0)[mask].mean()):.3f} of the AOI")

    df = L.draw_sample(cons, gb, N_PER_CLASS, s["block_km"] * 1000)
    # A class with almost no unambiguous pixels cannot be trained or tested;
    # it is dropped from this site's scheme (e.g. bare ground in Ljubljana).
    counts = df["y"].value_counts()
    dropped = counts[counts < MIN_CLASS_N].index.tolist()
    if dropped:
        print(f"Dropping classes with < {MIN_CLASS_N} points: "
              f"{[CLASSES[k] for k in dropped]}")
        df = df[~df["y"].isin(dropped)].reset_index(drop=True)
    sgk = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    fold = np.zeros(len(df), int)
    for i, (_, te) in enumerate(sgk.split(df, df["y"], df["block"])):
        fold[te] = i
    df["fold"] = fold
    df["split"] = np.where(df["fold"] < TEST_FOLDS, "test", "train")
    df.to_csv(s["data"] / "sample_points.csv", index=False)
    summ = df.groupby(["y", "split"]).size().unstack(fill_value=0)
    summ["blocks"] = df.groupby("y")["block"].nunique()
    summ.index = [CLASSES[i] for i in summ.index]
    summ.to_csv(s["tables"] / "sample_design.csv")
    print(summ)
    with open(s["tables"] / "consensus_agreement.txt", "w") as f:
        f.write(f"wc2021_vs_io2021_agreement={agree:.4f}\n"
                f"consensus_aoi_fraction={float((cons > 0)[mask].mean()):.4f}\n")


if __name__ == "__main__":
    main()
