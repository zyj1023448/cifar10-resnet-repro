"""Validate the committed artefacts without retraining anything."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf8"))


def main() -> int:
    errors = []
    groups = SUMMARY["groups"]
    for name, info in groups.items():
        for rid in info["runs"]:
            w = ROOT / "weights" / f"{rid}.pt"
            p = ROOT / "results" / "preds" / f"{rid}.npz"
            if not w.exists():
                errors.append(f"missing weight {w.name}")
                continue
            if not p.exists():
                errors.append(f"missing preds {p.name}")
                continue
            d = np.load(p)
            acc = float((d["logits"].argmax(1) == d["labels"]).mean())
            ck = torch.load(w, map_location="cpu", weights_only=False)
            logged = float(ck["metrics"]["test_acc"])
            if abs(acc - logged) > 1e-6:
                errors.append(f"{rid}: preds acc {acc:.6f} != logged {logged:.6f}")
    figs = list((ROOT / "results" / "figures").glob("*.png"))
    if len(figs) < 8:
        errors.append(f"only {len(figs)} figures")
    print(f"groups={len(groups)} runs={sum(len(v['runs']) for v in groups.values())} figures={len(figs)}")
    if errors:
        print("FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("PASS: all weights, predictions and figures are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
