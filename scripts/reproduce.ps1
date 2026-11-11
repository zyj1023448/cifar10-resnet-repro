# Reproduce all experiments from a clean checkout (Windows PowerShell).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python scripts/prepare_cifar.py
python scripts/run_grid.py
python -m src.analyze
python paper/build_paper.py
