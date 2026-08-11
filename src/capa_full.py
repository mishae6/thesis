import json
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from enrich import format_features, _cache_key

ROOT = SRC_DIR.parent
FEATURES = ROOT / "data" / "features_full.jsonl"
CAPA_CACHE = ROOT / "cache" / "capa"
CAPA_CACHE.mkdir(parents=True, exist_ok=True)
STATS = CAPA_CACHE / "_stats.jsonl"

ZIP = Path("/Volumes/Bodmas/BODMAS_disarmed_malware_binaries.zip")
RULES = str(ROOT / "capa-rules-9.4.0")
SIGS = str(ROOT / "capa-sigs")

MAGIC_TO_MACHINE = {0x10b: 0x014c, 0x20b: 0x8664}  # PE32->x86, PE32+->x64
PACKER_LIMIT_RULE = "(internal) packer file limitation"


def restore_machine(data):
    """Restore the PE machine field that BODMAS disarming zeroed.

    Returns repaired bytes, or None if architecture cannot be determined.
    """
    b = bytearray(data)
    try:
        e = struct.unpack_from("<I", b, 0x3C)[0]
        magic = struct.unpack_from("<H", b, e + 24)[0]
    except struct.error:
        return None
    machine = MAGIC_TO_MACHINE.get(magic)
    if machine is None:
        return None
    struct.pack_into("<H", b, e + 4, machine)
    return bytes(b)


def run_capa(data):
    with tempfile.NamedTemporaryFile(suffix=".exe", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        r = subprocess.run(
            ["capa", "-r", RULES, "-s", SIGS, "-j", tmp.name],
            capture_output=True, text=True,
        )
    return r.returncode, r.stdout, r.stderr


def capa_description(doc):
    """Build the CAPA arm's text: '[namespace] rule name' per matched rule.

    Returns (text, n_rules, hit_packer_limit).
    """
    lines = []
    hit_packer_limit = False
    for name, rule in doc.get("rules", {}).items():
        ns = rule.get("meta", {}).get("namespace", "")
        lines.append(f"[{ns}] {name}" if ns else name)
        if name == PACKER_LIMIT_RULE:
            hit_packer_limit = True
    lines.sort()
    return "\n".join(lines), len(lines), hit_packer_limit


def main():
    if not ZIP.exists():
        sys.exit(f"ZIP not found at {ZIP} - is the SSD mounted?")

    records = [json.loads(l) for l in open(FEATURES)]
    total = len(records)
    print(f"{total} records. Caching CAPA descriptions by feature-text hash.")

    zf = zipfile.ZipFile(ZIP)
    zip_members = set(zf.namelist())

    done = 0
    ran = 0
    skipped_cached = 0
    failures = []
    stats_fh = open(STATS, "a")
    t0 = time.time()

    seen_keys = set()

    for rec in records:
        done += 1
        key = _cache_key(format_features(rec))
        cache_file = CAPA_CACHE / f"{key}.txt"

        # Decision 1a: one CAPA run per unique feature-text hash.
        if cache_file.exists() or key in seen_keys:
            skipped_cached += 1
            continue
        seen_keys.add(key)

        sha = rec["sha256"]
        member = f"altered/{sha}.exe"
        if member not in zip_members:
            failures.append({"sha256": sha, "error": "not in zip"})
            continue

        data = zf.read(member)
        fixed = restore_machine(data)
        if fixed is None:
            failures.append({"sha256": sha, "error": "arch undetermined"})
            continue

        rc, out, err = run_capa(fixed)
        if rc != 0:
            failures.append({"sha256": sha, "error": f"capa rc={rc}: {err[-200:]}"})
            continue
        try:
            doc = json.loads(out)
        except json.JSONDecodeError:
            failures.append({"sha256": sha, "error": "capa output not JSON"})
            continue

        text, n_rules, hit_limit = capa_description(doc)
        cache_file.write_text(text)
        ran += 1

        stats_fh.write(json.dumps({
            "key": key, "sha256": sha, "category": rec["category"],
            "n_rules": n_rules, "packer_limited": hit_limit,
        }) + "\n")
        stats_fh.flush()

        if ran % 25 == 0:
            elapsed = time.time() - t0
            rate = elapsed / ran
            # remaining unique keys still to run is unknown exactly; estimate
            print(f"ran {ran} unique | {rate:.1f}s/run | {done}/{total} scanned")

    stats_fh.close()
    print(f"\nDone. Scanned {done}/{total}. CAPA runs: {ran}. "
          f"Skipped (cached/dup): {skipped_cached}.")
    if failures:
        print(f"Failures ({len(failures)}):")
        for f in failures[:20]:
            print(f"  {f['sha256'][:12]}: {f['error']}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()