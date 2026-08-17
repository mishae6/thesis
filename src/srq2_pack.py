import json
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capa_full import restore_machine

ROOT = Path(__file__).resolve().parent.parent
UNPACKED_LIST = ROOT / "results" / "srq2_unpacked_testsplit.jsonl"
ZIP_PATH = Path("/Volumes/Bodmas/BODMAS_disarmed_malware_binaries.zip")
WORK = Path("/Volumes/Bodmas/srq2_work")
ORIG_DIR = WORK / "orig"
PACKED_DIR = WORK / "packed"
LOG = ROOT / "results" / "srq2_pack_log.jsonl"

UPX = "/opt/homebrew/bin/upx"


def restore_subsystem(data, value=3):
    """BODMAS disarming also zeroes the PE subsystem field; UPX refuses
    subsystem 0. Restore a valid value (3 = console) so the packer runs.
    Returns repaired bytes, or None if the header can't be read."""
    b = bytearray(data)
    try:
        e = struct.unpack_from("<I", b, 0x3C)[0]
        struct.pack_into("<H", b, e + 24 + 0x44, value)
    except struct.error:
        return None
    return bytes(b)


def main():
    ORIG_DIR.mkdir(parents=True, exist_ok=True)
    PACKED_DIR.mkdir(parents=True, exist_ok=True)

    shas = [json.loads(line)["sha256"] for line in open(UNPACKED_LIST)]
    print(f"samples to pack: {len(shas)}")

    log_rows = []
    ok = fail_extract = fail_restore = fail_pack = skipped = 0

    with zipfile.ZipFile(ZIP_PATH) as z:
        names = set(z.namelist())
        for i, sha in enumerate(shas, 1):
            member = f"altered/{sha}.exe"
            orig_path = ORIG_DIR / f"{sha}.exe"
            packed_path = PACKED_DIR / f"{sha}.exe"

            if packed_path.exists():
                skipped += 1
                continue

            if member not in names:
                fail_extract += 1
                log_rows.append({"sha256": sha, "status": "missing_in_zip"})
                continue

            try:
                raw = z.open(member).read()
            except Exception as e:
                fail_extract += 1
                log_rows.append({"sha256": sha, "status": "extract_error", "detail": str(e)[:120]})
                continue

            # Restore the two header fields BODMAS disarming zeroed
            repaired = restore_machine(raw)
            if repaired is not None:
                repaired = restore_subsystem(repaired)
            if repaired is None:
                fail_restore += 1
                log_rows.append({"sha256": sha, "status": "restore_failed"})
                continue

            with open(orig_path, "wb") as dst:
                dst.write(repaired)

            try:
                r = subprocess.run(
                    [UPX, "-9", "-o", str(packed_path), str(orig_path)],
                    capture_output=True, text=True, timeout=120,
                )
                if r.returncode == 0 and packed_path.exists():
                    ok += 1
                    log_rows.append({"sha256": sha, "status": "packed"})
                else:
                    fail_pack += 1
                    msg = (r.stderr or r.stdout or "").strip().splitlines()
                    log_rows.append({"sha256": sha, "status": "upx_refused",
                                     "detail": (msg[-1] if msg else "")[:160]})
            except subprocess.TimeoutExpired:
                fail_pack += 1
                log_rows.append({"sha256": sha, "status": "upx_timeout"})

            try:
                orig_path.unlink()
            except OSError:
                pass

            if i % 50 == 0:
                print(f"  {i}/{len(shas)}  ok={ok} refused={fail_pack} "
                      f"restore_fail={fail_restore} missing={fail_extract} skip={skipped}")

    with open(LOG, "w") as f:
        for row in log_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\ndone. packed={ok}  upx_refused={fail_pack}  restore_failed={fail_restore}  "
          f"missing/extract_fail={fail_extract}  skipped={skipped}")
    print(f"packed binaries in: {PACKED_DIR}")
    print(f"log: {LOG}")


if __name__ == "__main__":
    main()