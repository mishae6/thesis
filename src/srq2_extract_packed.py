import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import extract_features       # identical extractor used for originals
from extract_full import _clean            # identical bytes-to-str cleaning

ROOT = Path(__file__).resolve().parent.parent
PACKED_DIR = Path("/Volumes/Bodmas/srq2_work/packed")
UNPACKED_LIST = ROOT / "results" / "srq2_unpacked_testsplit.jsonl"
OUTPUT = ROOT / "data" / "features_packed.jsonl"


def load_done():
    done = set()
    if OUTPUT.exists():
        for line in open(OUTPUT):
            done.add(json.loads(line)["sha256"])
    return done


def main():
    # category lookup for each packed sha (from the 8.2 list)
    cat_by_sha = {json.loads(l)["sha256"]: json.loads(l)["category"]
                  for l in open(UNPACKED_LIST)}

    packed_files = sorted(PACKED_DIR.glob("*.exe"))
    print(f"packed binaries found: {len(packed_files)}")

    done = load_done()
    if done:
        print(f"resuming: {len(done)} already extracted, skipping them")

    ok = fail = 0
    with open(OUTPUT, "a") as out:
        for i, path in enumerate(packed_files, 1):
            sha = path.stem  # filename is <sha256>.exe
            if sha in done:
                continue
            try:
                feats = extract_features(str(path))
            except Exception as e:
                fail += 1
                print(f"  extract failed {sha[:12]}: {str(e)[:80]}")
                continue
            if feats is None:
                fail += 1
                continue

            feats = _clean(feats)
            feats["sha256"] = sha
            feats["category"] = cat_by_sha.get(sha)
            feats.pop("path", None)  # drop the stale temp path field

            out.write(json.dumps(feats) + "\n")
            out.flush()
            ok += 1

            if i % 50 == 0:
                print(f"  {i}/{len(packed_files)}  ok={ok} fail={fail}")

    print(f"\ndone. extracted={ok}  failed={fail}")
    print(f"written: {OUTPUT}")


if __name__ == "__main__":
    main()