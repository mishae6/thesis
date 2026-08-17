import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNPACKED = ROOT / "results" / "srq2_unpacked_testsplit.jsonl"
LOG = ROOT / "results" / "srq2_pack_log.jsonl"

CATEGORIES = ["backdoor", "downloader", "dropper", "informationstealer",
              "ransomware", "trojan", "virus", "worm"]


def main():
    cat_by_sha = {json.loads(l)["sha256"]: json.loads(l)["category"] for l in open(UNPACKED)}
    status_by_sha = {json.loads(l)["sha256"]: json.loads(l)["status"] for l in open(LOG)}

    packed = Counter()
    refused = Counter()
    for sha, cat in cat_by_sha.items():
        st = status_by_sha.get(sha)
        if st == "packed":
            packed[cat] += 1
        else:
            refused[cat] += 1

    print("category            packed / (packed+refused)")
    for c in CATEGORIES:
        tot = packed[c] + refused[c]
        print(f"  {c:20s} {packed[c]:4d} / {tot:4d}")
    print(f"\ntotal packed: {sum(packed.values())}")


if __name__ == "__main__":
    main()