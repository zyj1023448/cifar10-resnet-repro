"""Extra depth experiment: SE-ResNet-56, started after the main grid finishes."""
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
        if time.time() - t0 > 4 * 3600:
            print("timeout waiting for grid", flush=True)
            return
        time.sleep(30)
    for seed in (42, 43):
        tag = f"se56_s{seed}"
        out = ROOT / "results" / "runs" / f"{tag}.json"
        if out.exists():
            print("skip", tag, flush=True)
            continue
        cmd = [PY, "-m", "src.train", "--model", "se_resnet56", "--tag", tag,
               "--seed", str(seed), "--epochs", "100", "--amp", "bf16"]
        print("RUN", " ".join(cmd), flush=True)
        with open(ROOT / "results" / "grid.log", "a", encoding="utf8") as log:
            log.write(f"\n===== {tag} {time.strftime('%H:%M:%S')} =====\n")
            log.flush()
            subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=False)
    print("EXTRA DONE", flush=True)


if __name__ == "__main__":
    main()
