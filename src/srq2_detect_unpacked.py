import ast
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_pairs, make_splits, CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features_full.jsonl"
OUT = ROOT / "results" / "srq2_unpacked_testsplit.jsonl"

PACKER_SECTION_SIGS = ("upx", "aspack", "petite", "nsp", "pklstb",
                       "mpress", "themida", ".vmp", ".enigma")
HIGH_ENTROPY = 7.5
CODE_SECTIONS = (".text", "code", ".itext")


def parse_field(raw):
    if isinstance(raw, (list, dict)):
        return raw
    if not raw:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None


def looks_packed(section_names, section_entropies):
    reasons = []
    for n in [str(x).lower() for x in (section_names or [])]:
        for sig in PACKER_SECTION_SIGS:
            if sig in n:
                reasons.append(f"packer_section:{n}")
    # Only trust "looks random" when it's the code section, not resources/icons
    for k, v in (section_entropies or {}).items():
        if not isinstance(v, (int, float)):
            continue
        if v >= HIGH_ENTROPY and any(cs in str(k).lower() for cs in CODE_SECTIONS):
            reasons.append(f"high_entropy_code:{k}={v}")
    return (len(reasons) > 0, reasons)


def main():
    pairs = load_pairs()
    *_, test_idx = make_splits(pairs)
    test_pairs = [pairs[i] for i in test_idx]
    print(f"test-split samples: {len(test_pairs)}")

    feats_by_sha = {}
    with open(FEATURES) as f:
        for line in f:
            r = json.loads(line)
            feats_by_sha[r["sha256"]] = r

    OUT.parent.mkdir(parents=True, exist_ok=True)
    unpacked, packed_reasons = [], []
    missing = 0
    per_cat_unpacked, per_cat_total = Counter(), Counter()

    for p in test_pairs:
        sha, cat = p.get("sha256"), p.get("category")
        per_cat_total[cat] += 1
        rec = feats_by_sha.get(sha)
        if rec is None:
            missing += 1
            continue
        names = parse_field(rec.get("section_names"))
        ents = parse_field(rec.get("section_entropies"))
        is_packed, reasons = looks_packed(names, ents)
        if is_packed:
            packed_reasons.append({"sha256": sha, "category": cat, "reasons": reasons})
        else:
            unpacked.append({"sha256": sha, "category": cat, "key": p.get("key")})
            per_cat_unpacked[cat] += 1

    with open(OUT, "w") as f:
        for u in unpacked:
            f.write(json.dumps(u) + "\n")
    with open(OUT.with_name("srq2_packed_excluded.jsonl"), "w") as f:
        for r in packed_reasons:
            f.write(json.dumps(r) + "\n")

    print(f"\nclassified: {len(test_pairs)}")
    print(f"  packed (excluded):  {len(packed_reasons)}")
    print(f"  unpacked (to pack): {len(unpacked)}")
    print(f"  missing features:   {missing}")
    print("\nunpacked / total per category:")
    for c in CATEGORIES:
        print(f"  {c:20s} {per_cat_unpacked[c]:4d} / {per_cat_total[c]:4d}")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()