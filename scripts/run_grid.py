"""Run the full experiment grid sequentially, skipping finished runs.

Each run is a separate process so that GPU state cannot leak between models.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(Path(sys.executable))
EPOCHS = 100

GRID = []


def add(tag, model, seeds, **kw):
    for s in seeds:
        GRID.append({"tag": f"{tag}_s{s}", "model": model, "seed": s, **kw})


# Main comparison (3 seeds)
add("plain20", "plain20", [42, 43, 44])
add("resnet20", "resnet20", [42, 43, 44])
add("se20", "se_resnet20", [42, 43, 44])

# Depth study (2 seeds)
add("plain56", "plain56", [42, 43])
add("resnet56", "resnet56", [42, 43])

# SE design ablation
add("se20_r4", "se_resnet20", [42, 43], se_reduction=4)
add("se20_r16", "se_resnet20", [42, 43], se_reduction=16)
add("se20_scope-stage3", "se_resnet20", [42], se_scope="stage3")
add("se20_scope-last", "se_resnet20", [42], se_scope="last")
add("preact20", "preact20", [42])

# Regularisation ablation on top of the proposed model
add("se20_mixup", "se_resnet20", [42, 43], mixup=0.2)
add("se20_ls", "se_resnet20", [42], label_smoothing=0.1)
add("se20_mixup-ls", "se_resnet20", [42, 43], mixup=0.2, label_smoothing=0.1)


def run(cfg: dict) -> None:
    tag = cfg.pop("tag")
    out = ROOT / "results" / "runs" / f"{tag}.json"
    if out.exists():
        try:
            rec = json.loads(out.read_text(encoding="utf8"))
            if rec["final"]["epochs"] == EPOCHS:
                print(f"skip {tag}", flush=True)
                return
        except Exception:
            pass
    cmd = [PY, "-m", "src.train", "--tag", tag, "--epochs", str(EPOCHS),
           "--amp", "bf16", "--model", cfg.pop("model"), "--seed", str(cfg.pop("seed"))]
    for k, v in cfg.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    print("RUN", " ".join(cmd), flush=True)
    t0 = time.time()
    with open(ROOT / "results" / "grid.log", "a", encoding="utf8") as log:
        log.write(f"\n===== {tag} {time.strftime('%H:%M:%S')} =====\n")
        log.flush()
        subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=False)
    print(f"done {tag} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    print(f"{len(GRID)} runs, {EPOCHS} epochs", flush=True)
    for cfg in GRID:
        run(dict(cfg))
    print("ALL DONE", flush=True)
