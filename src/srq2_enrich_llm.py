import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich import enrich_one_cached, format_features, _cache_key

ROOT = Path(__file__).resolve().parent.parent
PACKED_FEATURES = ROOT / "data" / "features_packed.jsonl"
OUTPUT = ROOT / "results" / "srq2_llm_packed.jsonl"


def load_done():
    done = set()
    if OUTPUT.exists():
        for line in open(OUTPUT):
            done.add(json.loads(line)["sha256"])
    return done


def main():
    records = [json.loads(l) for l in open(PACKED_FEATURES)]
    print(f"packed feature records: {len(records)}")

    done = load_done()
    if done:
        print(f"resuming: {len(done)} already enriched, skipping")

    ok = fail = 0
    with open(OUTPUT, "a") as out:
        for i, rec in enumerate(records, 1):
            sha = rec["sha256"]
            if sha in done:
                continue
            try:
                # frozen prompt, unchanged; only the input (packed record) differs
                description = enrich_one_cached(rec)
                if description is None:
                    # older cache func may not return; read back from cache
                    key = _cache_key(format_features(rec))
                    description = (ROOT / "cache" / "descriptions" / f"{key}.txt").read_text()
            except Exception as e:
                fail += 1
                print(f"  enrich failed {sha[:12]}: {str(e)[:80]}")
                continue

            out.write(json.dumps({
                "sha256": sha,
                "category": rec.get("category"),
                "key": _cache_key(format_features(rec)),
                "description": description,
            }) + "\n")
            out.flush()
            ok += 1

            if i % 25 == 0:
                print(f"  {i}/{len(records)}  ok={ok} fail={fail}")

    print(f"\ndone. enriched={ok}  failed={fail}")
    print(f"written: {OUTPUT}")


if __name__ == "__main__":
    main()