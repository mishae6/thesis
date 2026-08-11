import pandas as pd
from pathlib import Path

# Anchor paths to this script's location, so it runs from anywhere (same
# fix applied to enrich.py's cache path in Step 5).
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent                      # thesis/
CSV = ROOT / "data" / "bodmas_malware_category.csv"
MANIFEST = ROOT / "data" / "selection_manifest.csv"

# The 8 categories kept for classification; the other 6 are too sparse.
CATEGORIES = ["backdoor", "downloader", "dropper", "informationstealer",
              "ransomware", "trojan", "virus", "worm"]

CAP = 2000        # maximum samples taken from any one category
SEED = 42         # fixes the random draw so selection is reproducible


def main():
    df = pd.read_csv(CSV)

    # Keep only the 8 target categories, discard the rest.
    df = df[df["category"].isin(CATEGORIES)]

    chosen = []
    for cat in CATEGORIES:
        rows = df[df["category"] == cat]
        if len(rows) > CAP:
            # Category exceeds the cap: draw CAP samples at random, seeded.
            rows = rows.sample(n=CAP, random_state=SEED)
        # else: category is at or under the cap, take every row.
        chosen.append(rows)

    manifest = pd.concat(chosen).reset_index(drop=True)

    # Report counts per category so they can be eyeballed against the design
    # targets before anything is extracted.
    print("Selected samples per category:")
    print(manifest["category"].value_counts().sort_index())
    print(f"\nTotal selected: {len(manifest)}")

    manifest.to_csv(MANIFEST, index=False)
    print(f"Manifest written to: {MANIFEST}")


if __name__ == "__main__":
    main()