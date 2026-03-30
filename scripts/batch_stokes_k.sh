#!/usr/bin/env bash
set -euo pipefail

# Batch runs for Stokes configs with k in {2,3,4,5}.
# Usage:
#   bash scripts/batch_stokes_k.sh
#   CONDA_ENV=div-free bash scripts/batch_stokes_k.sh

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
cfg.setdefault('basis', {})['k'] = kval
run = cfg.setdefault('run', {})
name = str(run.get('name', 'stokes_experiment'))
if re.search(r'k=\d+', name):
    name = re.sub(r'k=\d+', f'k={kval}', name)
else:
    name = f"{name}-k={kval}"
run['name'] = name
json.dump(cfg, open(out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
PY
}

if [[ -n "${BASE_CONFIGS_STR:-}" ]]; then
  IFS=':' read -r -a BASE_CONFIGS <<< "$BASE_CONFIGS_STR"
else
  BASE_CONFIGS=(
    "configs/stokes/2d_stokes_config.json"
    "configs/stokes/3d_stokes_config.json"
  )
fi
K_LIST=(2 3 4 5)

for base in "${BASE_CONFIGS[@]}"; do
  for k in "${K_LIST[@]}"; do
    tmp_cfg="$(mktemp /tmp/stokes_k${k}_XXXXXX.json)"
    update_cfg_k "$base" "$tmp_cfg" "$k"
    echo "[batch-stokes] $(date '+%F %T') run base=$(basename "$base") k=$k"
    run_py src/run_experiments_stokes.py "$tmp_cfg"
    rm -f "$tmp_cfg"
  done
done
