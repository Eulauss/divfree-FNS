#!/usr/bin/env bash
set -euo pipefail

run_py() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" python "$@"
  else
    python "$@"
  fi
}

tmp_py="$(mktemp /tmp/draw_l2_XXXXXX.py)"
cat > "$tmp_py" <<'PY'
import sys
sys.path.insert(0, "src")
import divfree.plotting.drawing_l2 as mod

mod.drawing_config["csv_paths"] = [
    "outputs/l2_approx/2d_L2/divfree2d-nomass-k=1/rate_table.csv",
    "outputs/l2_approx/2d_L2/divfree2d-nomass-k=2/rate_table.csv",
    "outputs/l2_approx/2d_L2/divfree2d-nomass-k=3/rate_table.csv",
    "outputs/l2_approx/2d_L2/divfree2d-nomass-k=4/rate_table.csv",
]
mod.drawing_config["out_dir"] = "outputs/figures_and_tables/l2_approx/2d_l2"
mod.drawing_config["fig_name"] = "L2_d=2-nomass.png"
mod.drawing_config["reference_line_config"] = {
    "1": {
        "slope": -0.75,
        "intercept": 15,
        "x_range": [400, 2800],
    },
    "2": {
        "slope": -1.25,
        "intercept": 40,
        "x_range": [400, 2800],
    },
    "3": {
        "slope": -1.75,
        "intercept": 90,
        "x_range": [400, 2800],
    },
    "4": {
        "slope": -2.25,
        "intercept": 260,
        "x_range": [400, 2800],
    },
}
mod.main()

mod.drawing_config["csv_paths"] = [
    "outputs/l2_approx/3d_L2/divfree3d-mass-k=1/rate_table.csv",
    "outputs/l2_approx/3d_L2/divfree3d-mass-k=2/rate_table.csv",
    "outputs/l2_approx/3d_L2/divfree3d-mass-k=3/rate_table.csv",
    "outputs/l2_approx/3d_L2/divfree3d-mass-k=4/rate_table.csv",
]
mod.drawing_config["out_dir"] = "outputs/figures_and_tables/l2_approx/3d_l2"
mod.drawing_config["fig_name"] = "L2_d=3-mass.png"
mod.drawing_config["reference_line_config"] = {
    "1": {
        "slope": -2/3,
        "intercept": 24,
        "x_range": [1000, 3500],
    },
    "2": {
        "slope": -1.0,
        "intercept": 50,
        "x_range": [1000, 3500],
    },
    "3": {
        "slope": -4/3,
        "intercept": 110,
        "x_range": [1000, 3500],
    },
    "4": {
        "slope": -5/3,
        "intercept": 400,
        "x_range": [1000, 3500],
    },
}
mod.main()
PY
run_py "$tmp_py"
rm -f "$tmp_py"
