#!/usr/bin/env bash
set -euo pipefail

run_py() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" python "$@"
  else
    python "$@"
  fi
}

tmp_py="$(mktemp /tmp/draw_stokes_XXXXXX.py)"
cat > "$tmp_py" <<'PY'
import sys
sys.path.insert(0, "src")
import divfree.plotting.drawing_stokes_v2 as mod

mod.drawing_config["csv_paths"] = [
    "outputs/stokes/2d_stokes/stokes2d-nomass-k=2/rate_table.csv",
    "outputs/stokes/2d_stokes/stokes2d-nomass-k=3/rate_table.csv",
    "outputs/stokes/2d_stokes/stokes2d-nomass-k=4/rate_table.csv",
    "outputs/stokes/2d_stokes/stokes2d-nomass-k=5/rate_table.csv",
]
mod.drawing_config["out_dir"] = "outputs/figures_and_tables/stokes/2d_stokes"
mod.drawing_config["fig_name"] = "stokes_d=2-nomass.png"
mod.drawing_config["reference_line_config"] = {
    "2": {
        "slope": -0.75,
        "intercept": 22,
        "x_range": [400, 2800],
    },
    "3": {
        "slope": -1.25,
        "intercept": 100,
        "x_range": [400, 2800],
    },
    "4": {
        "slope": -1.75,
        "intercept": 580,
        "x_range": [400, 2800],
    },
    "5": {
        "slope": -2.25,
        "intercept": 3800,
        "x_range": [400, 2800],
    },
}
mod.main()

mod.drawing_config["csv_paths"] = [
    "outputs/stokes/3d_stokes/stokes3d-mass-k=2/rate_table.csv",
    "outputs/stokes/3d_stokes/stokes3d-mass-k=3/rate_table.csv",
    "outputs/stokes/3d_stokes/stokes3d-mass-k=4/rate_table.csv",
    "outputs/stokes/3d_stokes/stokes3d-mass-k=5/rate_table.csv",
]
mod.drawing_config["out_dir"] = "outputs/figures_and_tables/stokes/3d_stokes"
mod.drawing_config["fig_name"] = "stokes_d=3-mass.png"
mod.drawing_config["reference_line_config"] = {
    "2": {
        "slope": -2/3,
        "intercept": 37,
        "x_range": [1200, 3500],
    },
    "3": {
        "slope": -1.0,
        "intercept": 150,
        "x_range": [1200, 3500],
    },
    "4": {
        "slope": -4/3,
        "intercept": 850,
        "x_range": [1200, 3500],
    },
    "5": {
        "slope": -5/3,
        "intercept": 5300,
        "x_range": [1200, 3500],
    },
}
mod.main()
PY
run_py "$tmp_py"
rm -f "$tmp_py"
