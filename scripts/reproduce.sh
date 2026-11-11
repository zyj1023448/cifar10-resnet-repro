#!/usr/bin/env bash
# Reproduce all experiments from a clean checkout (Linux/macOS with CUDA).
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare_cifar.py
python scripts/run_grid.py
python -m src.analyze
python paper/build_paper.py
