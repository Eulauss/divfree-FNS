#!/usr/bin/env bash
set -euo pipefail

# Single runs for FEM-vs-FNS configs (no k sweep).
# Usage:
#   bash scripts/batch_stokes_lid_femonly.sh
#   CONDA_ENV=div-free bash scripts/batch_stokes_lid_femonly.sh

run_py() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" python "$@"
  else
    python "$@"
  fi
}

if [[ -n "${BASE_CONFIGS_STR:-}" ]]; then
  IFS=':' read -r -a BASE_CONFIGS <<< "$BASE_CONFIGS_STR"
else
  BASE_CONFIGS=(
    "configs/stokes_lid/2d_stokes_lid_femonly_classic_config.json"
    "configs/stokes_lid/2d_stokes_lid_femonly_cos2_config.json"
  )
fi

for cfg in "${BASE_CONFIGS[@]}"; do
  echo "[batch-femonly] $(date '+%F %T') run $(basename "$cfg")"
  run_py src/run_fem_vs_fns.py --config "$cfg"
done
