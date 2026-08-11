"""
extract_full.py — Phase 2 of the full-scale training-set build.

Reads data/selection_manifest.csv (the 9,207 chosen samples), extracts each
binary's static features from the zip one at a time (extract, read, delete),
and writes one feature record per line to data/features_full.jsonl.

Each output record is the extract_features() dict plus the sha256 and category
from the manifest, so downstream code can pair features to labels.

Resumable: if features_full.jsonl already contains records, samples already
done are skipped, so an interrupted run can be restarted safely.
"""

import json
import zipfile
import tempfile
import os
from pathlib import Path

import pandas as pd

from extract import extract_features   # reuse the Step 3 LIEF extractor

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
MANIFEST = ROOT / "data" / "selection_manifest.csv"
OUTPUT = ROOT / "data" / "features_full.jsonl"
ZIP_PATH = "/Volumes/Bodmas/BODMAS_disarmed_malware_binaries.zip"

# Files inside the zip live under this prefix, named <sha256>.exe
ZIP_PREFIX = "altered/"


def _clean(obj):
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {_clean(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    return obj


def load_done():
    """Return the set of sha256 already present in the output file."""
    done = set()
    if OUTPUT.exists():
        for line in open(OUTPUT):
            rec = json.loads(line)
            done.add(rec["sha256"])
    return done


def main():
    manifest = pd.read_csv(MANIFEST)
    done = load_done()
    if done:
        print(f"Resuming: {len(done)} samples already extracted, skipping them.")

    total = len(manifest)
    written = 0
    parse_failures = []

    # Open the zip once and keep it open for the whole run.
    with zipfile.ZipFile(ZIP_PATH) as zf, open(OUTPUT, "a") as out:
        for i, row in manifest.iterrows():
            sha = row["sha256"]
            category = row["category"]

            if sha in done:
                continue

            entry = f"{ZIP_PREFIX}{sha}.exe"

            # Extract this one file's bytes from the zip.
            try:
                data = zf.read(entry)
            except KeyError:
                # Manifest hash not found in the zip: a fatal inconsistency,
                # stop and investigate rather than silently skip.
                raise RuntimeError(f"Sample not found in zip: {entry}")

            tmp = tempfile.NamedTemporaryFile(suffix=".exe", delete=False)
            try:
                tmp.write(data)
                tmp.close()

                features = extract_features(tmp.name)
            finally:
                os.unlink(tmp.name)   # always delete the temp binary

            if features is None:
                # LIEF could not parse this binary: an expected rare event
                # at scale, log which sample and continue.
                parse_failures.append(sha)
                continue

            # Attach identity and label from the manifest.
            features["sha256"] = sha
            features["category"] = category

            out.write(json.dumps(_clean(features)) + "\n")
            written += 1

            if written % 100 == 0:
                print(f"{written} written ({i + 1}/{total} scanned)")

    print(f"\nDone. {written} new records written to {OUTPUT}")
    if parse_failures:
        print(f"LIEF parse failures ({len(parse_failures)}): {parse_failures}")


if __name__ == "__main__":
    main()