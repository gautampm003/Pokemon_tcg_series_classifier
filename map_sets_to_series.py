"""
Map local Pokémon TCG dataset folders (one per set) to their parent series
using the TCGdex REST API, then build a label CSV for training.

Run this locally (this environment has no internet access to TCGdex).

Usage:
    pip install requests rapidfuzz
    python map_sets_to_series.py "C:\\Users\\HP\\Documents\\ACM\\Pokemon_CNN\\Pokemon TCG\\Pokemon TCG" "C:\\Users\\HP\\Documents\\ACM\\Pokemon_CNN"

    Pass as many root directories as you have. Every immediate subfolder of
    EVERY root passed is treated as one "set folder" (e.g. the ~196 sets
    inside the nested "Pokemon TCG\\Pokemon TCG" folder, PLUS the 5 extra
    set folders like 30th-celebration that sit one level up). If the same
    folder name exists under two roots, they are counted as the same set
    and their image counts are summed - only expect that if you deliberately
    duplicated something.

Output:
    dataset_metadata.csv       -- set,series,image_count for every matched folder
    unmatched_folders.csv      -- folders we could NOT confidently match, for manual review
    series_summary.csv         -- aggregated image count per series (the class balance table)
"""

import os
import sys
import csv
import requests
from rapidfuzz import process, fuzz

TCGDEX_SETS_URL = "https://api.tcgdex.net/v2/en/sets"

# Manual overrides for folder names that won't fuzzy-match cleanly.
# Add to this as you review unmatched_folders.csv.
MANUAL_OVERRIDES = {
    "151": "Scarlet & Violet",
    "base-set": "Base",
    "base-set-2": "Base",

    "black-white-energy-2011-unnumbered": "Black & White",
    "black-white-promos": "Black & White",
    "black-white-trainer-kit-excadrill": "Black & White",
    "black-white-trainer-kit-zoroark": "Black & White",

    "diamond-pearl-promos": "Diamond & Pearl",
    "diamond-pearl-trainer-kit-lucario": "Diamond & Pearl",
    "diamond-pearl-trainer-kit-manaphy": "Diamond & Pearl",

    "hs-energy-2010-unnumbered": "HeartGold & SoulSilver",

    "pokemon-futsal-promos-2020": "Sword & Shield",

    "scarlet-violet-promos": "Scarlet & Violet",

    "sun-moon-promos": "Sun & Moon",
    "sun-moon-energy": "Sun & Moon",
    "sun-moon": "Sun & Moon",

    "sword-shield-energy": "Sword & Shield",
    "sword-shield-promos": "Sword & Shield",

    "xy-energy-2013-unnumbered": "XY",

    # Leave these out for now
    "mee": None,
    "miscellaneous": None,
    "box-topper": None,
    "victory-medals": None,
    "world-collection": None,
    "mcdonalds-match-battle-2023": None,

    "pokemon-trading-card-game-classic-blastoise": None,
    "pokemon-trading-card-game-classic-charizard": None,
    "pokemon-trading-card-game-classic-venusaur": None,
}

FUZZY_MATCH_THRESHOLD = 82  # 0-100, tune down if too many misses, up if false matches appear


def slugify(name: str) -> str:
    return (
        name.lower()
        .replace("&", "and")
        .replace("'", "")
        .replace(":", "")
        .replace(",", "")
        .replace(".", "")
        .replace("  ", " ")
        .strip()
        .replace(" ", "-")
    )


def fetch_tcgdex_sets():
    print("Fetching set list from TCGdex...")
    resp = requests.get(TCGDEX_SETS_URL, timeout=30)
    resp.raise_for_status()
    sets = resp.json()
    print(f"  Got {len(sets)} sets from TCGdex.\n")
    return sets


def fetch_series_for_set(set_id: str):
    """TCGdex's /sets list endpoint doesn't always inline series; fetch the
    individual set record which includes its parent series."""
    url = f"https://api.tcgdex.net/v2/en/sets/{set_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        series = data.get("serie", {}).get("name")
        return series
    except Exception as e:
        print(f"    Warning: could not fetch series for set {set_id}: {e}")
        return None


def count_images(folder_path: str) -> int:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    count = 0
    for _, _, files in os.walk(folder_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                count += 1
    return count


def main():
    if len(sys.argv) < 2:
        print('Usage: python map_sets_to_series.py "path/to/root1" ["path/to/root2" ...]')
        sys.exit(1)

    roots = sys.argv[1:]
    for r in roots:
        if not os.path.isdir(r):
            print(f"Not a directory: {r}")
            sys.exit(1)

    # folder_name -> list of full paths (usually 1, but support duplicates across roots)
    # Also skip any subfolder that is itself one of the OTHER roots passed in
    # (e.g. "Pokemon_CNN" contains a "Pokemon TCG" folder that's already being
    # scanned directly as its own root - don't double-count its ~20k images).
    root_basenames = {os.path.basename(os.path.normpath(r)) for r in roots}

    folder_paths = {}
    for root in roots:
        for d in sorted(os.listdir(root)):
            full_path = os.path.join(root, d)
            if not os.path.isdir(full_path):
                continue
            if d in root_basenames:
                print(f"  Skipping '{full_path}' - it's a container for another root, not a set folder.")
                continue
            folder_paths.setdefault(d, []).append(full_path)

    local_folders = sorted(folder_paths.keys())
    print(f"Found {len(local_folders)} distinct set folders across {len(roots)} root(s).\n")

    tcgdex_sets = fetch_tcgdex_sets()
    # Build lookup: slugified TCGdex set name -> set id
    tcgdex_lookup = {slugify(s["name"]): s["id"] for s in tcgdex_sets}
    tcgdex_names = list(tcgdex_lookup.keys())

    # Cache series lookups by TCGdex set id so we don't refetch for duplicates
    series_cache = {}

    matched_rows = []
    unmatched = []

    for folder in local_folders:
        img_count = sum(count_images(p) for p in folder_paths[folder])

        # 1. Manual override takes priority
        if folder in MANUAL_OVERRIDES:
            series = MANUAL_OVERRIDES[folder]
            if series is None:
                unmatched.append((folder, img_count, "manual_override_needs_review"))
            else:
                matched_rows.append((folder, series, img_count))
            continue

        # 2. Try fuzzy match against TCGdex set names
        best_match, score, _ = process.extractOne(
            folder, tcgdex_names, scorer=fuzz.WRatio
        )
        if score >= FUZZY_MATCH_THRESHOLD:
            set_id = tcgdex_lookup[best_match]
            if set_id not in series_cache:
                series_cache[set_id] = fetch_series_for_set(set_id)
            series = series_cache[set_id]
            if series:
                matched_rows.append((folder, series, img_count))
            else:
                unmatched.append((folder, img_count, f"matched set '{best_match}' but no series found"))
        else:
            unmatched.append((folder, img_count, f"best guess '{best_match}' score={score}"))

    # Write dataset_metadata.csv
    with open("dataset_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["set_folder", "series", "image_count"])
        writer.writerows(matched_rows)
    print(f"\nWrote dataset_metadata.csv ({len(matched_rows)} matched folders)")

    # Write unmatched_folders.csv for manual review
    with open("unmatched_folders.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["set_folder", "image_count", "reason"])
        writer.writerows(unmatched)
    print(f"Wrote unmatched_folders.csv ({len(unmatched)} folders need manual review)")

    # Aggregate series_summary.csv
    series_totals = {}
    for _, series, count in matched_rows:
        series_totals[series] = series_totals.get(series, 0) + count
    with open("series_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "total_images"])
        for series, total in sorted(series_totals.items(), key=lambda x: -x[1]):
            writer.writerow([series, total])
    print("Wrote series_summary.csv (class balance table)\n")

    print("=== Series distribution (highest to lowest) ===")
    for series, total in sorted(series_totals.items(), key=lambda x: -x[1]):
        flag = "  <-- LOW, consider merging/excluding" if total < 500 else ""
        print(f"  {series}: {total}{flag}")

    if unmatched:
        print(f"\n{len(unmatched)} folders were NOT matched. Check unmatched_folders.csv")
        print("and add entries to MANUAL_OVERRIDES at the top of this script, then rerun.")


if __name__ == "__main__":
    main()