"""One extra seed for the pre-activation ablation, started after the grid."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(Path(sys.executable))
LOG = ROOT.parent / ".tmp" / "grid_stdout.log"


def grid_done() -> bool:
    try:
        return "ALL DONE" in LOG.read_text(encoding="utf8", errors="ignore")
    except FileNotFoundError:
        return False


def main() -> None:
    t0 = time.time()
    while not grid_done():
        if time.time() - t0 > 3 * 3600:
            print("timeout", flush=True)
            return
        time.sleep(20)
    tag = "preact20_s43"
    if (ROOT / "results" / "runs" / f"{tag}.json").exists():
        print("skip", tag, flush=True)
        return
    cmd = [PY, "-m", "src.train", "--model", "preact20", "--tag", tag,
           "--seed", "43", "--epochs", "100", "--amp", "bf16"]
    with open(ROOT / "results" / "grid.log", "a", encoding="utf8") as log:
        log.write(f"\n===== {tag} {time.strftime('%H:%M:%S')} =====\n")
        log.flush()
        subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=False)
    print("done", tag, flush=True)


if __name__ == "__main__":
    main()
