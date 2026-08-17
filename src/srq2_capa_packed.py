import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capa_full import run_capa, capa_description
from enrich import format_features, _cache_key

ROOT = Path(__file__).resolve().parent.parent
PACKED_FEATURES = ROOT / "data" / "features_packed.jsonl"
PACKED_DIR = Path("/Volumes/Bodmas/srq2_work/packed")
OUTPUT = ROOT / "results" / "srq2_capa_packed.jsonl"


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
        print(f"resuming: {len(done)} already done, skipping")

    ok = fail = 0
    failures = []
    with open(OUTPUT, "a") as out:
        for i, rec in enumerate(records, 1):
            sha = rec["sha256"]
            if sha in done:
                continue

            packed_path = PACKED_DIR / f"{sha}.exe"
            if not packed_path.exists():
                fail += 1
                failures.append({"sha256": sha, "error": "packed file missing"})
                continue

            data = packed_path.read_bytes()  # already machine+subsystem restored in 8.3
            rc, stdout, stderr = run_capa(data)
            if rc != 0:
                fail += 1
                failures.append({"sha256": sha, "error": f"capa rc={rc}: {stderr[-150:]}"})
                continue

            try:
                doc = json.loads(stdout)
                text, n_rules, hit_limit = capa_description(doc)
            except Exception as e:
                fail += 1
                failures.append({"sha256": sha, "error": f"parse: {str(e)[:100]}"})
                continue

            out.write(json.dumps({
                "sha256": sha,
                "category": rec.get("category"),
                "key": _cache_key(format_features(rec)),
                "description": text,
                "n_rules": n_rules,
                "hit_packer_limit": hit_limit,
            }) + "\n")
            out.flush()
            ok += 1

            if i % 25 == 0:
                print(f"  {i}/{len(records)}  ok={ok} fail={fail}")

    with open(OUTPUT.with_name("srq2_capa_packed_failures.jsonl"), "w") as f:
        for row in failures:
            f.write(json.dumps(row) + "\n")

    print(f"\ndone. capa_ok={ok}  failed={fail}")
    print(f"written: {OUTPUT}")


if __name__ == "__main__":
    main()