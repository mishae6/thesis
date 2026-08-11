# smoke_capa.py — verify CAPA installs, parses disarmed BODMAS binaries, time it, dump its JSON
import json, subprocess, sys, tempfile, time, zipfile
from pathlib import Path

ZIP = Path("/Volumes/Bodmas/BODMAS_disarmed_malware_binaries.zip")
RULES = "capa-rules-9.4.0"
SIGS = "capa-sigs"
N = 3

def run_capa(path):
    t0 = time.time()
    r = subprocess.run(["capa", "-r", RULES, "-s", SIGS, "-j", str(path)], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr, time.time() - t0

def main():
    if not ZIP.exists():
        print(f"ZIP not found at {ZIP} — is the SSD mounted? (ls /Volumes/Bodmas)")
        sys.exit(1)
    zf = zipfile.ZipFile(ZIP)
    members = [m for m in zf.namelist() if m.lower().endswith(".exe")][:N]
    print(f"Testing {len(members)} binaries from {ZIP.name}\n")
    for i, m in enumerate(members):
        data = zf.read(m)
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            rc, out, err, secs = run_capa(tmp.name)
        print(f"[{i}] {m}  ({len(data)} bytes)  rc={rc}  {secs:.1f}s")
        if rc != 0:
            print("  STDERR (last 800 chars):")
            print("  " + err[-800:].replace("\n", "\n  "))
            continue
        try:
            doc = json.loads(out)
        except json.JSONDecodeError:
            print("  rc=0 but stdout not JSON; first 400 chars:\n  " + out[:400])
            continue
        rules = doc.get("rules", {})
        print(f"  parsed OK; {len(rules)} matched rules")
        if i == 0:
            Path("capa_sample.json").write_text(out)
            print("  wrote full JSON to capa_sample.json")

if __name__ == "__main__":
    main()