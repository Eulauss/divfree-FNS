#!/usr/bin/env bash
set -euo pipefail

# Batch runs for lid-driven cavity configs with k in {2,3}.
# Usage:
#   bash scripts/batch_stokes_lid_k.sh
#   CONDA_ENV=div-free bash scripts/batch_stokes_lid_k.sh

run_py() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" python "$@"
  else
    python "$@"
  fi
}

update_cfg_k() {
  local base_cfg="$1"
  local out_cfg="$2"
  local kval="$3"
  python3 - "$base_cfg" "$out_cfg" "$kval" <<'PY'
import json
import re
import sys

base, out, kval = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = json.load(open(base, 'r', encoding='utf-8'))
cfg.setdefault('fns', {})['reluk_k'] = kval
is_mass = (kval >= 3)
cfg.setdefault('fns', {})['mass_matrix'] = is_mass
name = str(cfg.get('experiment_name', 'stokes_lid'))
if re.search(r'k=\d+', name):
    name = re.sub(r'k=\d+', f'k={kval}', name)
else:
    name = f"{name}-k={kval}"
if re.search(r'(?:^|-)nomass(?:-|$)', name):
    name = re.sub(r'nomass', 'mass' if is_mass else 'nomass', name)
elif re.search(r'(?:^|-)mass(?:-|$)', name):
    name = re.sub(r'mass', 'mass' if is_mass else 'nomass', name)
else:
    name = f"{name}-{'mass' if is_mass else 'nomass'}"
cfg['experiment_name'] = name
json.dump(cfg, open(out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
PY
}

if [[ -n "${BASE_CONFIGS_STR:-}" ]]; then
  IFS=':' read -r -a BASE_CONFIGS <<< "$BASE_CONFIGS_STR"
else
  BASE_CONFIGS=(
    "configs/stokes_lid/2d_stokes_lid_classic_config.json"
    "configs/stokes_lid/2d_stokes_lid_cos2_config.json"
  )
fi
K_LIST=(2 3)

for base in "${BASE_CONFIGS[@]}"; do
  for k in "${K_LIST[@]}"; do
    tmp_cfg="$(mktemp /tmp/stokes_lid_k${k}_XXXXXX.json)"
    update_cfg_k "$base" "$tmp_cfg" "$k"
    echo "[batch-lid] $(date '+%F %T') run base=$(basename "$base") k=$k"
    run_py src/run_experiments_lid.py --config "$tmp_cfg"
    rm -f "$tmp_cfg"
  done
done
