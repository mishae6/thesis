import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "data" / "features_full.jsonl"
LLM_CACHE = ROOT / "cache" / "descriptions"
CAPA_CACHE = ROOT / "cache" / "capa"

import sys
sys.path.insert(0, str(ROOT / "src"))
from enrich import format_features, _cache_key

CATEGORIES = ["backdoor", "downloader", "dropper", "informationstealer",
              "ransomware", "trojan", "virus", "worm"]

# per-category F1: unenriched and LLM, from the SRQ1 results (for the correlation)
F1_UNENRICHED = {"backdoor":0.951,"downloader":0.890,"dropper":0.778,
                 "informationstealer":0.281,"ransomware":0.764,"trojan":0.529,
                 "virus":1.000,"worm":0.581}
F1_LLM = {"backdoor":0.954,"downloader":0.747,"dropper":0.042,
          "informationstealer":0.222,"ransomware":0.451,"trojan":0.427,
          "virus":0.952,"worm":0.534}


def all_imports(rec):
    """Flat list of every imported function name in the record."""
    out = []
    for dll, funcs in rec.get("imports", {}).items():
        for f in funcs:
            if f:
                out.append(f)
    return out


def import_retention(rec, description):
    """Fraction of exact import names that appear (case-insensitive substring)
    in the description. Exact-token measure, not semantic."""
    imps = all_imports(rec)
    if not imps:
        return None  # no imports to retain
    desc_low = description.lower()
    kept = sum(1 for f in imps if f.lower() in desc_low)
    return kept, len(imps)


def mentions_any(names, description):
    """True if any of the given names appears in the description."""
    desc_low = description.lower()
    return any(n and n.lower() in desc_low for n in names)


def main():
    records = [json.loads(l) for l in open(FEATURES)]

    # dedupe to unique feature-texts (same keying as the arms)
    seen = set()
    unique = []
    for rec in records:
        key = _cache_key(format_features(rec))
        if key in seen:
            continue
        seen.add(key)
        unique.append((key, rec))

    print(f"{len(records)} samples, {len(unique)} unique feature-texts\n")

    # accumulate per category
    ret_kept = defaultdict(int)      # imports retained
    ret_total = defaultdict(int)     # imports present
    n_with_imports = defaultdict(int)
    n_desc = defaultdict(int)
    section_mentioned = defaultdict(int)  # descriptions mentioning any section name
    entropy_mentioned = defaultdict(int)  # descriptions mentioning any entropy value
    missing_llm = 0

    for key, rec in unique:
        cat = rec["category"]
        llm_file = LLM_CACHE / f"{key}.txt"
        if not llm_file.exists():
            missing_llm += 1
            continue
        desc = llm_file.read_text()
        n_desc[cat] += 1

        r = import_retention(rec, desc)
        if r is not None:
            kept, total = r
            ret_kept[cat] += kept
            ret_total[cat] += total
            n_with_imports[cat] += 1

        # section names present in record
        sec_names = rec.get("section_names", [])
        if mentions_any(sec_names, desc):
            section_mentioned[cat] += 1

        # entropy values (rounded, as strings) present in description
        ents = [str(round(v, 2)) for v in rec.get("section_entropies", {}).values()]
        if mentions_any(ents, desc):
            entropy_mentioned[cat] += 1

    print("=== LLM exact-token retention and structural-field loss (per category) ===")
    print("(retention = fraction of exact import NAMES surviving into the description;")
    print(" section/entropy = % of descriptions mentioning ANY section name / entropy value)\n")
    header = f"{'category':20s} {'imp_retain%':>11s} {'sect_ment%':>11s} {'ent_ment%':>10s} {'F1_drop':>8s}"
    print(header)
    print("-" * len(header))

    rows = []
    for cat in CATEGORIES:
        retain = 100 * ret_kept[cat] / ret_total[cat] if ret_total[cat] else float("nan")
        secpct = 100 * section_mentioned[cat] / n_desc[cat] if n_desc[cat] else float("nan")
        entpct = 100 * entropy_mentioned[cat] / n_desc[cat] if n_desc[cat] else float("nan")
        f1drop = F1_UNENRICHED[cat] - F1_LLM[cat]
        rows.append((cat, retain, secpct, entpct, f1drop))
        print(f"{cat:20s} {retain:>10.1f} {secpct:>10.1f} {entpct:>9.1f} {f1drop:>+8.3f}")

    # overall
    tot_kept = sum(ret_kept.values()); tot_total = sum(ret_total.values())
    overall = 100 * tot_kept / tot_total if tot_total else float("nan")
    print("-" * len(header))
    print(f"{'OVERALL':20s} {overall:>10.1f}")
    if missing_llm:
        print(f"\n(note: {missing_llm} unique feature-texts had no LLM description)")

    # correlation: does low retention track high F1 drop?
    print("\n=== Correlation: import-retention vs F1 drop (LLM vs unenriched) ===")
    valid = [(r[1], r[4]) for r in rows if r[1] == r[1]]  # drop NaNs
    if len(valid) >= 2:
        import statistics
        xs = [v[0] for v in valid]; ys = [v[1] for v in valid]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x-mx)*(y-my) for x, y in valid) / len(valid)
        sx = statistics.pstdev(xs); sy = statistics.pstdev(ys)
        r = cov / (sx*sy) if sx and sy else float("nan")
        print(f"Pearson r = {r:.3f}  (negative = lower retention -> larger F1 drop)")

    # dump to results/
    out = ROOT / "results" / "underperformance_analysis.txt"
    with open(out, "w") as fh:
        fh.write("LLM underperformance analysis (SRQ1)\n")
        fh.write("Exact import-name retention and structural-field loss, per category,\n")
        fh.write(f"measured across {len(unique)} unique feature-texts.\n\n")
        fh.write(header + "\n")
        for cat, retain, secpct, entpct, f1drop in rows:
            fh.write(f"{cat:20s} {retain:>10.1f} {secpct:>10.1f} {entpct:>9.1f} {f1drop:>+8.3f}\n")
        fh.write(f"\nOVERALL import retention: {overall:.1f}%\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()