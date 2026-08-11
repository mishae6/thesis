# capa_probe.py — restore machine field on a few disarmed BODMAS binaries,
# run CAPA for real, and show what the CAPA arm would feed the model.
import json, struct, subprocess, tempfile, time, zipfile
from pathlib import Path

ZIP = Path("/Volumes/Bodmas/BODMAS_disarmed_malware_binaries.zip")
RULES = "capa-rules-9.4.0"
SIGS = "capa-sigs"
N = 5

# optional-header magic -> machine word
MAGIC_TO_MACHINE = {0x10b: 0x014c, 0x20b: 0x8664}  # PE32->x86, PE32+->x64

def restore_machine(data: bytes) -> bytes:
    b = bytearray(data)
    e = struct.unpack_from("<I", b, 0x3C)[0]      # PE header offset
    magic = struct.unpack_from("<H", b, e + 24)[0]  # optional header magic
    machine = MAGIC_TO_MACHINE.get(magic)
    if machine is None:
        return None  # can't determine arch; caller logs and skips
    struct.pack_into("<H", b, e + 4, machine)      # write machine field
    return bytes(b)

def run_capa(data: bytes):
    with tempfile.NamedTemporaryFile(suffix=".exe", delete=True) as tmp:
        tmp.write(data); tmp.flush()
        t0 = time.time()
        r = subprocess.run(["capa", "-r", RULES, "-s", SIGS, "-j", tmp.name],
                           capture_output=True, text=True)
        return r.returncode, r.stdout, r.stderr, time.time() - t0

def capabilities(doc: dict):
    # returns list of (namespace, rule_name) for each matched rule
    out = []
    for name, rule in doc.get("rules", {}).items():
        ns = rule.get("meta", {}).get("namespace", "")
        out.append((ns, name))
    return sorted(out)

def main():
    zf = zipfile.ZipFile(ZIP)
    exes = [x for x in zf.namelist() if x.lower().endswith(".exe")][:N]
    for m in exes:
        data = zf.read(m)
        fixed = restore_machine(data)
        if fixed is None:
            print(f"\n{m[-16:]}  SKIP: could not determine architecture")
            continue
        rc, out, err, secs = run_capa(fixed)
        print(f"\n=== {m[-16:]}  rc={rc}  {secs:.1f}s ===")
        if rc != 0:
            print("  CAPA error, stderr tail:")
            print("  " + err[-400:].replace("\n", "\n  "))
            continue
        try:
            doc = json.loads(out)
        except json.JSONDecodeError:
            print("  rc=0 but not JSON; first 300 chars:\n  " + out[:300])
            continue
        caps = capabilities(doc)
        print(f"  {len(caps)} matched rules")
        for ns, name in caps:
            print(f"    [{ns}] {name}")

if __name__ == "__main__":
    main()