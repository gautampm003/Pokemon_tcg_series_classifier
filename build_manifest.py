"""
Build a per-image manifest for the Pokémon TCG series-classification project.

Reads dataset_metadata.csv (set_folder, series, image_count) produced by
map_sets_to_series.py, walks each matched set folder, and writes one row
per image: (image_path, series).

Also does a lightweight front/back sanity check: samples a handful of images
per folder and reports basic stats (aspect ratio, average brightness) so you
can spot anything that looks structurally different (e.g. card backs, which
in real Pokémon TCG products are visually IDENTICAL across every card/series -
if any snuck in, they'd be unlearnable noise and must be excluded).

Run this locally.

Usage:
    pip install pillow
    python build_manifest.py "dataset_metadata.csv" "C:\\Users\\HP\\Documents\\ACM\\Pokemon_CNN\\Pokemon TCG\\Pokemon TCG" "C:\\Users\\HP\\Documents\\ACM\\Pokemon_CNN"

    Pass dataset_metadata.csv first, then the same root directories you gave
    map_sets_to_series.py (in any order; the script searches all of them for
    each set_folder name).

Output:
    manifest.csv                -- image_path,series  (one row per image)
    manifest_sanity_report.csv  -- per-folder sample stats for eyeballing
"""

import os
import sys
import csv
import random
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Run: pip install pillow")
    sys.exit(1)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SAMPLE_PER_FOLDER = 5  # how many images per set folder to sample for the sanity report


def find_set_folder(set_folder_name: str, roots: list[str]) -> str | None:
    for root in roots:
        candidate = os.path.join(root, set_folder_name)
        if os.path.isdir(candidate):
            return candidate
    return None


def list_images(folder: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(dirpath, f))
    return paths


def image_stats(path: str):
    try:
        with Image.open(path) as img:
            w, h = img.size
            aspect = round(w / h, 3) if h else None
            # cheap average brightness on a downsized grayscale copy
            small = img.convert("L").resize((32, 32))
            pixels = list(small.getdata())
            brightness = round(sum(pixels) / len(pixels), 1)
            return w, h, aspect, brightness
    except Exception as e:
        return None, None, None, f"ERROR: {e}"


def main():
    if len(sys.argv) < 3:
        print('Usage: python build_manifest.py dataset_metadata.csv "root1" ["root2" ...]')
        sys.exit(1)

    metadata_path = sys.argv[1]
    roots = sys.argv[2:]
    for r in roots:
        if not os.path.isdir(r):
            print(f"Not a directory: {r}")
            sys.exit(1)

    set_to_series = {}
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            set_to_series[row["set_folder"]] = row["series"]

    print(f"Loaded {len(set_to_series)} set->series mappings from {metadata_path}\n")

    manifest_rows = []
    sanity_rows = []
    missing_folders = []

    for set_folder, series in set_to_series.items():
        folder_path = find_set_folder(set_folder, roots)
        if not folder_path:
            missing_folders.append(set_folder)
            continue

        images = list_images(folder_path)
        for img_path in images:
            manifest_rows.append((img_path, series))

        # sample a few for the sanity report
        sample = random.sample(images, min(SAMPLE_PER_FOLDER, len(images)))
        for img_path in sample:
            w, h, aspect, brightness = image_stats(img_path)
            sanity_rows.append((set_folder, series, os.path.basename(img_path), w, h, aspect, brightness))

    with open("manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "series"])
        writer.writerows(manifest_rows)
    print(f"Wrote manifest.csv ({len(manifest_rows)} image rows)")

    with open("manifest_sanity_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["set_folder", "series", "filename", "width", "height", "aspect_ratio", "avg_brightness"])
        writer.writerows(sanity_rows)
    print(f"Wrote manifest_sanity_report.csv ({len(sanity_rows)} sampled rows)\n")

    if missing_folders:
        print(f"WARNING: {len(missing_folders)} set folders from dataset_metadata.csv were not found on disk:")
        for m in missing_folders:
            print(f"  {m}")
        print()

    # Quick aggregate check: aspect ratio should cluster tightly if these are
    # all standard-format cards. A wildly different aspect ratio or a distinct
    # brightness cluster among a folder's samples is the signal to go look
    # at those specific images by eye (card backs, promos, oddly cropped scans).
    aspects = [row[5] for row in sanity_rows if isinstance(row[5], float)]
    if aspects:
        aspects_sorted = sorted(aspects)
        print(f"Aspect ratio range across sample: {aspects_sorted[0]} to {aspects_sorted[-1]}")
        print("Standard Pokémon card aspect ratio is ~0.718 (63mm x 88mm).")
        print("If you see values far from that (e.g. near 1.0, or > 1.0), open those")
        print("specific rows in manifest_sanity_report.csv and view the images -")
        print("they may be card backs, energy/promo art, or mis-scanned images.\n")

    print("Next: open manifest_sanity_report.csv, sort by aspect_ratio and by")
    print("avg_brightness, and skim the outliers as actual images before trusting")
    print("the full manifest for training.")


if __name__ == "__main__":
    main()