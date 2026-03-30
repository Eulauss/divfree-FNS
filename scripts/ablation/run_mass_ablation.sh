#!/usr/bin/env bash
set -euo pipefail

# Mass ablation runner + plotting script.
# It runs 2 configs and then draws mass-ablation figures.
#
# NOTE: This script can be heavy. Run on a proper compute node.
# Usage:
#   CONDA_ENV=div-free bash scripts/ablation/run_mass_ablation.sh

run_py() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" python "$@"
  else
    python "$@"
  fi
}

CFG_MASS="configs/mass_ablation/divfree2d-mass-k=4_config.json"
CFG_NOMASS="configs/mass_ablation/divfree2d-nomass=k=4_config.json"

echo "[mass-ablation] run $CFG_MASS"
run_py src/run_experiments.py "$CFG_MASS"

echo "[mass-ablation] run $CFG_NOMASS"
run_py src/run_experiments.py "$CFG_NOMASS"

echo "[mass-ablation] draw figures"
run_py src/divfree/plotting/drawing_mass_ablation.py

echo "[mass-ablation] done"
