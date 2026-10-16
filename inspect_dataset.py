"""
Pokémon TCG dataset inspector.

Run this AFTER downloading/unzipping the Kaggle dataset
(ellimaaac/pokemon-tcg-all-image-cards) locally.

Usage:
    python inspect_dataset.py /path/to/unzipped/dataset

What it does:
  1. Walks the folder tree and reports total image count + a sample of paths,
     so you can see how the data is actually organized (folder-per-set?
     flat with ID-based filenames? something else?).
  2. Looks for any bundled metadata (csv/json) and, if found, loads it and
     prints per-series and per-set counts.
  3. If NO bundled metadata is found, tries to infer set/series from
     filenames or folder names as a fallback, and tells you clearly that
     you'll need to join against TCGdex/pokemontcg.io data yourself.
  4. Flags classes (series) with very low image counts so you know what to
     exclude or merge before training.
"""

import os
import sys
import json
import glob
from collections import Counter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def walk_images(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(dirpath, f))
    return paths


def find_metadata_files(root):
    candidates = []
    for ext in ("csv", "json", "jsonl", "parquet"):
        candidates.extend(glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True))
    return candidates


def try_load_csv_metadata(path):
    import csv
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"  Could not parse {path} as CSV: {e}")
        return None
    return rows


def try_load_json_metadata(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"  Could not parse {path} as JSON: {e}")
        return None


def guess_series_field(sample_row):
    """Look for likely field names holding series/set info."""
    candidates = []
    for key in sample_row.keys():
        lk = key.lower()
        if "series" in lk or "set" in lk:
            candidates.append(key)
    return candidates


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_dataset.py /path/to/dataset")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        sys.exit(1)

    print(f"Scanning {root} ...\n")

    images = walk_images(root)
    print(f"Total image files found: {len(images)}")
    if images:
        print("Sample paths:")
        for p in images[:10]:
            print(" ", os.path.relpath(p, root))
    print()

    # Folder-based grouping (in case images are organized in per-set subfolders)
    top_level_dirs = Counter()
    for p in images:
        rel = os.path.relpath(p, root)
        parts = rel.split(os.sep)
        if len(parts) > 1:
            top_level_dirs[parts[0]] += 1
    if top_level_dirs:
        print("Top-level folder breakdown (possible set/series grouping):")
        for name, count in top_level_dirs.most_common(20):
            print(f"  {name}: {count} images")
        print()

    meta_files = find_metadata_files(root)
    print(f"Metadata files found: {len(meta_files)}")
    for m in meta_files[:10]:
        print(" ", os.path.relpath(m, root))
    print()

    if not meta_files:
        print("No bundled metadata found.")
        print("You'll need to join image filenames against TCGdex or")
        print("pokemontcg.io card IDs yourself to get series/set labels.")
        return

    # Try the first CSV or JSON metadata file found
    csv_files = [m for m in meta_files if m.endswith(".csv")]
    json_files = [m for m in meta_files if m.endswith(".json") or m.endswith(".jsonl")]

    rows = None
    if csv_files:
        print(f"Loading {csv_files[0]} ...")
        rows = try_load_csv_metadata(csv_files[0])
    elif json_files:
        print(f"Loading {json_files[0]} ...")
        data = try_load_json_metadata(json_files[0])
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            # common shape: {"cards": [...]}
            for key in ("cards", "data", "items"):
                if key in data and isinstance(data[key], list):
                    rows = data[key]
                    break

    if not rows:
        print("Found metadata files but couldn't parse rows automatically.")
        print("Open them manually to inspect the schema.")
        return

    print(f"\nLoaded {len(rows)} metadata rows.")
    sample = rows[0]
    print("Sample row keys:", list(sample.keys()))

    series_fields = guess_series_field(sample)
    print(f"Likely series/set fields: {series_fields}\n")

    for field in series_fields:
        counts = Counter()
        for row in rows:
            val = row.get(field)
            if isinstance(val, dict):
                val = val.get("name") or val.get("series") or str(val)
            counts[val] += 1
        print(f"--- Distribution by '{field}' ---")
        for val, count in counts.most_common(30):
            flag = "  <-- LOW COUNT, consider excluding/merging" if count < 200 else ""
            print(f"  {val}: {count}{flag}")
        print()


if __name__ == "__main__":
    main()