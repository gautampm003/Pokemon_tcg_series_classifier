"""
Stratified train/val/test split from manifest.csv.

Splits by series so every class keeps roughly the same proportion across
train/val/test - critical here since class sizes range from 88 to 3679.
No images are copied or moved; we just write three CSVs of paths+labels
and let PyTorch's Dataset read straight from the original folders.

Run this locally.

Usage:
    py -3.11 -m pip install scikit-learn pandas
    py -3.11 split_dataset.py manifest.csv

Output:
    train.csv, val.csv, test.csv   -- each with columns: image_path, series
    label_map.json                 -- series name -> integer class index,
                                       needed so the CNN trains on numbers,
                                       not strings, and so predictions can
                                       be mapped back to series names later
"""

import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15  # must sum to 1.0 with the above
RANDOM_SEED = 42  # fixed so the split is reproducible across reruns


def main():
    if len(sys.argv) < 2:
        print("Usage: py -3.11 split_dataset.py manifest.csv")
        sys.exit(1)

    manifest_path = sys.argv[1]
    df = pd.read_csv(manifest_path)
    print(f"Loaded {len(df)} images across {df['series'].nunique()} series.\n")

    # First split off the training set, then split the remainder into val/test.
    # stratify= ensures every series keeps its proportion in each split.
    train_df, remainder_df = train_test_split(
        df,
        train_size=TRAIN_FRAC,
        stratify=df["series"],
        random_state=RANDOM_SEED,
    )
    # remainder_df is (VAL_FRAC + TEST_FRAC) of the total; split it in half
    val_relative_frac = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_df, test_df = train_test_split(
        remainder_df,
        train_size=val_relative_frac,
        stratify=remainder_df["series"],
        random_state=RANDOM_SEED,
    )

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print(f"train.csv: {len(train_df)} images")
    print(f"val.csv:   {len(val_df)} images")
    print(f"test.csv:  {len(test_df)} images\n")

    # Build and save the label map (series name -> integer index), sorted
    # alphabetically so it's deterministic across reruns.
    series_list = sorted(df["series"].unique())
    label_map = {series: idx for idx, series in enumerate(series_list)}
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"Wrote label_map.json ({len(label_map)} classes)\n")

    # Sanity check: print per-series counts in each split so you can eyeball
    # that stratification actually worked (proportions should look similar
    # across the three columns for every series).
    print("=== Per-series counts (train / val / test) ===")
    for series in series_list:
        t = (train_df["series"] == series).sum()
        v = (val_df["series"] == series).sum()
        te = (test_df["series"] == series).sum()
        print(f"  {series:30s} {t:5d} / {v:5d} / {te:5d}")


if __name__ == "__main__":
    main()