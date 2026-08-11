import json
import time
from pathlib import Path

from enrich import enrich_one_cached

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
FEATURES = ROOT / "data" / "features_full.jsonl"


def main():
    records = [json.loads(l) for l in open(FEATURES)]
    total = len(records)
    print(f"Enriching {total} samples.")

    done = 0
    failures = []
    t0 = time.time()

    for i, rec in enumerate(records, 1):
        try:
            enrich_one_cached(rec)
        except Exception as e:
            failures.append({"sha256": rec.get("sha256"), "error": str(e)})

        done += 1
        if done % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / done
            remaining = (total - done) * rate / 3600
            print(f"{done}/{total} | {rate:.1f}s/sample | ~{remaining:.1f}h left")

    print(f"\nDone. Processed {done}/{total}.")
    if failures:
        print(f"Failures ({len(failures)}):")
        for f in failures:
            print(f"  {f['sha256']}: {f['error']}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()