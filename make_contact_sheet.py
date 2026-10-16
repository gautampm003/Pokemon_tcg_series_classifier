"""
Build one or more contact-sheet images (grids of thumbnails) sampled across
manifest.csv, so you can eyeball for card backs / anomalies in a few seconds
per sheet instead of opening images one at a time.

Run this locally.

Usage:
    pip install pillow
    python make_contact_sheet.py manifest.csv

Output:
    contact_sheet_01.jpg, contact_sheet_02.jpg, ... in the current folder.
    Each sheet is a grid of ~100 thumbnails with the series name printed
    under each one, randomly sampled (stratified so every series appears
    at least a few times across the sheets).
"""
print("CONTACT SHEET SCRIPT STARTED")
import sys
import csv
import random
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

THUMB_SIZE = (120, 168)  # matches ~0.714 aspect ratio
GRID_COLS = 10
GRID_ROWS = 10
PER_SHEET = GRID_COLS * GRID_ROWS
LABEL_HEIGHT = 14


def main():
    if len(sys.argv) < 2:
        print("Usage: python make_contact_sheet.py manifest.csv")
        sys.exit(1)

    manifest_path = sys.argv[1]
    by_series = defaultdict(list)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_series[row["series"]].append(row["image_path"])

    # Stratified sample: pull a proportional slice from every series so rare
    # classes still show up on the sheets, not just the huge ones.
    total_target = 300  # spread across 3 sheets of 100
    total_images = sum(len(v) for v in by_series.values())
    sampled = []
    for series, paths in by_series.items():
        n = max(3, round(total_target * len(paths) / total_images))
        n = min(n, len(paths))
        chosen = random.sample(paths, n)
        sampled.extend((p, series) for p in chosen)

    random.shuffle(sampled)
    print(f"Sampled {len(sampled)} images across {len(by_series)} series for contact sheets.")

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    num_sheets = (len(sampled) + PER_SHEET - 1) // PER_SHEET
    for sheet_idx in range(num_sheets):
        chunk = sampled[sheet_idx * PER_SHEET : (sheet_idx + 1) * PER_SHEET]
        sheet_w = GRID_COLS * THUMB_SIZE[0]
        sheet_h = GRID_ROWS * (THUMB_SIZE[1] + LABEL_HEIGHT)
        sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
        draw = ImageDraw.Draw(sheet)

        for i, (path, series) in enumerate(chunk):
            row, col = divmod(i, GRID_COLS)
            x = col * THUMB_SIZE[0]
            y = row * (THUMB_SIZE[1] + LABEL_HEIGHT)
            try:
                with Image.open(path) as img:
                    thumb = img.convert("RGB").resize(THUMB_SIZE)
                sheet.paste(thumb, (x, y))
            except Exception as e:
                draw.rectangle([x, y, x + THUMB_SIZE[0], y + THUMB_SIZE[1]], fill="red")
                print(f"  Could not open {path}: {e}")

            label = series[:16]
            draw.text((x + 2, y + THUMB_SIZE[1] + 1), label, fill="black", font=font)

        out_name = f"contact_sheet_{sheet_idx+1:02d}.jpg"
        sheet.save(out_name, quality=85)
        print(f"Wrote {out_name} ({len(chunk)} thumbnails)")

    print("\nOpen each contact_sheet_*.jpg and scan for:")
    print("  - any image that looks like a card BACK (same blue Poké Ball")
    print("    design repeated identically regardless of the series label)")
    print("  - blank/corrupt thumbnails (shown as solid red if unreadable)")
    print("  - anything that isn't a Pokémon card at all (energy cards, box art, etc.)")


if __name__ == "__main__":
    main()