#!/usr/bin/env bash
set -euo pipefail

# Boundary ablation runner + config_used sync script.
# It runs 6 Stokes configs (3 for 2D nomass, 3 for 3D mass)
# and then syncs config_used.json under outputs/boundary_ablation/*.
#
# NOTE: This script can be heavy. Run on a proper compute node.
# Usage:
#   CONDA_ENV=div-free bash scripts/ablation/run_boundary_ablation.sh

run_py() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" python "$@"
  else
    python "$@"
  fi
}

CFG_2D=(
  "configs/boundary_ablation/stokes2d-nomass/2d_stokes_config-lambdry=0.01.json"
  "configs/boundary_ablation/stokes2d-nomass/2d_stokes_config-lambdry=1.json"
  "configs/boundary_ablation/stokes2d-nomass/2d_stokes_config-lambdry=100.json"
)

CFG_3D=(
  "configs/boundary_ablation/stokes3d-mass/3d_stokes_config-lambdry=0.06.json"
  "configs/boundary_ablation/stokes3d-mass/3d_stokes_config-lambdry=6.json"
  "configs/boundary_ablation/stokes3d-mass/3d_stokes_config-lambdry=600.json"
)

for cfg in "${CFG_2D[@]}"; do
  echo "[boundary-ablation] run $cfg"
  run_py src/run_experiments_stokes.py "$cfg"
done

for cfg in "${CFG_3D[@]}"; do
  echo "[boundary-ablation] run $cfg"
  run_py src/run_experiments_stokes.py "$cfg"
done

# Sync config_used.json by parsing each config's output metadata.
sync_cfg_used() {
  local cfg="$1"
  python3 - "$cfg" <<'PY'
import json
import shutil
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
run = cfg.get("run", {})
problem = cfg.get("problem", {})
exp = cfg.get("experiment", {})

out_root = Path(run.get("out_dir", "outputs/boundary_ablation"))
run_name = run.get("name")
ptype = str(problem.get("type", "stokes2d"))
mass = bool(exp.get("mass_matrix", False))
group = f"{ptype}-{'mass' if mass else 'nomass'}"

if not run_name:
    raise ValueError(f"Missing run.name in {cfg_path}")

dst = out_root / group / run_name / "config_used.json"
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(cfg_path, dst)
print(f"[boundary-ablation] synced {cfg_path} -> {dst}")
PY
}

for cfg in "${CFG_2D[@]}"; do
  sync_cfg_used "$cfg"
done

for cfg in "${CFG_3D[@]}"; do
  sync_cfg_used "$cfg"
done

echo "[boundary-ablation] done"
