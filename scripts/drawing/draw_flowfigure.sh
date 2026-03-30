#!/usr/bin/env bash
set -euo pipefail

run_py() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" python "$@"
  else
    python "$@"
  fi
}

tmp_py="$(mktemp /tmp/draw_flow_XXXXXX.py)"
cat > "$tmp_py" <<'PY'
import sys
sys.path.insert(0, "src")
import divfree.plotting.drawing_flowfigure as mod

mod.FLOWFIG_CONFIG["target_family"] = "l2_divfree2d"
mod.FLOWFIG_CONFIG["target_kind"] = "stream_sin"
mod.FLOWFIG_CONFIG["ckpt_paths"] = [
    "outputs/l2_approx/2d_L2/divfree2d-nomass-k=2/check_points/divfree2d_M=26.npz",
    "outputs/l2_approx/2d_L2/divfree2d-nomass-k=2/check_points/divfree2d_M=51.npz",
    "outputs/l2_approx/2d_L2/divfree2d-nomass-k=2/check_points/divfree2d_M=202.npz",
]
mod.FLOWFIG_CONFIG["out_dir"] = "outputs/flowfigures"
mod.FLOWFIG_CONFIG["out_basename"] = "flowfield_gt_pred_err-d2k2-nomass-nopositive.png"
mod.main()

mod.FLOWFIG_CONFIG["target_family"] = "stokes2d"
mod.FLOWFIG_CONFIG["ckpt_paths"] = [
    "outputs/stokes/2d_stokes/stokes2d-nomass-k=2/check_points/stokes2d_M=26.npz",
    "outputs/stokes/2d_stokes/stokes2d-nomass-k=2/check_points/stokes2d_M=100.npz",
    "outputs/stokes/2d_stokes/stokes2d-nomass-k=2/check_points/stokes2d_M=400.npz",
]
mod.FLOWFIG_CONFIG["out_dir"] = "outputs/flowfigures"
mod.FLOWFIG_CONFIG["out_basename"] = "flowfield_gt_pred_err-stokes-d2k2-nomass-nopositive.png"
mod.main()
PY
run_py "$tmp_py"
rm -f "$tmp_py"
