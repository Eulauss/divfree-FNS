#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for all main experiments, drawings, and ablations.
# Real-time terminal output + log file in project root.
#
# Usage:
#   CONDA_ENV=div-free bash scripts/run_all_experiments.sh
#   EXPERIMENT_SCALE=small CONDA_ENV=div-free bash scripts/run_all_experiments.sh
#   DRY_RUN=1 CONDA_ENV=div-free bash scripts/run_all_experiments.sh
#
# Optional envs:
#   LOG_FILE=run_all_YYYYmmdd_HHMMSS.log
#   DRY_RUN=1      # print steps only, do not execute
#   SKIP_ABLATION=1
#   EXPERIMENT_SCALE=full|small

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_FILE="${LOG_FILE:-run_all_$(date +%Y%m%d_%H%M%S).log}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_ABLATION="${SKIP_ABLATION:-0}"
EXPERIMENT_SCALE="${EXPERIMENT_SCALE:-full}"

if [[ "$EXPERIMENT_SCALE" != "full" && "$EXPERIMENT_SCALE" != "small" ]]; then
  echo "[run-all] invalid EXPERIMENT_SCALE=$EXPERIMENT_SCALE (expected full or small)" >&2
  exit 1
fi

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[run-all] start: $(date '+%F %T')"
echo "[run-all] root: $ROOT_DIR"
echo "[run-all] log:  $LOG_FILE"
echo "[run-all] CONDA_ENV=${CONDA_ENV:-<unset>}"
echo "[run-all] DRY_RUN=$DRY_RUN SKIP_ABLATION=$SKIP_ABLATION EXPERIMENT_SCALE=$EXPERIMENT_SCALE"

TMP_CFG_DIR=""

cleanup() {
  if [[ -n "$TMP_CFG_DIR" && -d "$TMP_CFG_DIR" ]]; then
    rm -rf "$TMP_CFG_DIR"
  fi
}

trap cleanup EXIT

join_by_colon() {
  local first="$1"
  shift
  printf '%s' "$first"
  local item
  for item in "$@"; do
    printf ':%s' "$item"
  done
}

make_small_cfg() {
  local base_cfg="$1"
  local out_cfg="$2"
  local cfg_kind="$3"
  python3 - "$base_cfg" "$out_cfg" "$cfg_kind" <<'PY'
import json
import sys

base_cfg, out_cfg, cfg_kind = sys.argv[1], sys.argv[2], sys.argv[3]
with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = json.load(f)

if cfg_kind in {"l2", "stokes"}:
    exp = cfg.setdefault("experiment", {})
    exp["M_list"] = [m for m in exp.get("M_list", []) if m <= 512]
elif cfg_kind == "stokes_lid":
    fns = cfg.setdefault("fns", {})
    if "n_list" in fns:
        fns["n_list"] = [n for n in fns.get("n_list", []) if n <= 512]
    ref = cfg.get("reference")
    if isinstance(ref, dict) and "fem_mesh_N" in ref:
        ref["fem_mesh_N"] = 64
    fem_compare = cfg.get("fem_compare")
    if isinstance(fem_compare, dict):
        if "reference_mesh_N" in fem_compare:
            fem_compare["reference_mesh_N"] = 64
        if "mesh_list" in fem_compare:
            fem_compare["mesh_list"] = [m for m in fem_compare.get("mesh_list", []) if m in {2, 3, 4, 6, 9}]
else:
    raise ValueError(f"Unknown cfg kind: {cfg_kind}")

with open(out_cfg, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
}

L2_CMD=(bash scripts/batch_l2_k.sh)
STOKES_CMD=(bash scripts/batch_stokes_k.sh)
LID_CMD=(bash scripts/batch_stokes_lid_k.sh)
FEMONLY_CMD=(bash scripts/batch_stokes_lid_femonly.sh)

if [[ "$EXPERIMENT_SCALE" == "small" ]]; then
  TMP_CFG_DIR="$(mktemp -d /tmp/run_all_small_cfgs_XXXXXX)"

  L2_CFGS=(
    "$TMP_CFG_DIR/2d_L2_config.json"
    "$TMP_CFG_DIR/3d_L2_config.json"
  )
  STOKES_CFGS=(
    "$TMP_CFG_DIR/2d_stokes_config.json"
    "$TMP_CFG_DIR/3d_stokes_config.json"
  )
  LID_CFGS=(
    "$TMP_CFG_DIR/2d_stokes_lid_classic_config.json"
    "$TMP_CFG_DIR/2d_stokes_lid_cos2_config.json"
  )
  FEMONLY_CFGS=(
    "$TMP_CFG_DIR/2d_stokes_lid_femonly_classic_config.json"
    "$TMP_CFG_DIR/2d_stokes_lid_femonly_cos2_config.json"
  )

  make_small_cfg "configs/l2_approx/2d_L2_config.json" "${L2_CFGS[0]}" "l2"
  make_small_cfg "configs/l2_approx/3d_L2_config.json" "${L2_CFGS[1]}" "l2"
  make_small_cfg "configs/stokes/2d_stokes_config.json" "${STOKES_CFGS[0]}" "stokes"
  make_small_cfg "configs/stokes/3d_stokes_config.json" "${STOKES_CFGS[1]}" "stokes"
  make_small_cfg "configs/stokes_lid/2d_stokes_lid_classic_config.json" "${LID_CFGS[0]}" "stokes_lid"
  make_small_cfg "configs/stokes_lid/2d_stokes_lid_cos2_config.json" "${LID_CFGS[1]}" "stokes_lid"
  make_small_cfg "configs/stokes_lid/2d_stokes_lid_femonly_classic_config.json" "${FEMONLY_CFGS[0]}" "stokes_lid"
  make_small_cfg "configs/stokes_lid/2d_stokes_lid_femonly_cos2_config.json" "${FEMONLY_CFGS[1]}" "stokes_lid"

  L2_CMD=(env "BASE_CONFIGS_STR=$(join_by_colon "${L2_CFGS[@]}")" bash scripts/batch_l2_k.sh)
  STOKES_CMD=(env "BASE_CONFIGS_STR=$(join_by_colon "${STOKES_CFGS[@]}")" bash scripts/batch_stokes_k.sh)
  LID_CMD=(env "BASE_CONFIGS_STR=$(join_by_colon "${LID_CFGS[@]}")" bash scripts/batch_stokes_lid_k.sh)
  FEMONLY_CMD=(env "BASE_CONFIGS_STR=$(join_by_colon "${FEMONLY_CFGS[@]}")" bash scripts/batch_stokes_lid_femonly.sh)

  echo "[run-all] small-scale configs created in $TMP_CFG_DIR"
fi

run_step() {
  local title="$1"
  shift
  echo
  echo "[run-all] ===== $title ====="
  echo "[run-all] cmd: $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

# Main experiments.
run_step "L2 sweep (2D/3D, k=1..4)" "${L2_CMD[@]}"
run_step "Stokes sweep (2D/3D, k=2..5)" "${STOKES_CMD[@]}"
run_step "Lid sweep (classic+cos2, k=2,3)" "${LID_CMD[@]}"
run_step "FEM-vs-FNS (classic+cos2)" "${FEMONLY_CMD[@]}"

# Drawing scripts (depend on outputs above).
run_step "Draw L2 figures" bash scripts/drawing/draw_l2.sh
run_step "Draw Stokes figures" bash scripts/drawing/draw_stokes.sh
run_step "Draw Lid convergence figures" bash scripts/drawing/draw_lid_convergence.sh
run_step "Draw flowfield figures" bash scripts/drawing/draw_flowfigure.sh

# Ablations (heavy).
if [[ "$SKIP_ABLATION" != "1" ]]; then
  run_step "Boundary ablation" bash scripts/ablation/run_boundary_ablation.sh
  run_step "Mass ablation (+ drawing)" bash scripts/ablation/run_mass_ablation.sh
else
  echo "[run-all] skip ablation by SKIP_ABLATION=1"
fi

echo
echo "[run-all] done: $(date '+%F %T')"
