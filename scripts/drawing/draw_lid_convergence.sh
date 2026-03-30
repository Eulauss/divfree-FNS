#!/usr/bin/env bash
set -euo pipefail

run_py() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" python "$@"
  else
    python "$@"
  fi
}

tmp_py="$(mktemp /tmp/draw_lid_conv_XXXXXX.py)"
cat > "$tmp_py" <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")
import divfree.plotting.drawing_convergence as mod


def merge_records(input_paths, out_json):
    all_records = []
    for p in input_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Missing input state file: {path}")
        data = json.loads(path.read_text())
        recs = data.get("records", []) if isinstance(data, dict) else data
        if not isinstance(recs, list):
            raise ValueError(f"Invalid records format: {path}")
        all_records.extend(recs)

    uniq = {}
    for r in all_records:
        key = json.dumps(r, sort_keys=True, ensure_ascii=False)
        uniq[key] = r
    merged = list(uniq.values())

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"records": merged}, indent=2, ensure_ascii=False))
    return out_path

cos2_json = merge_records(
    [
        "outputs/stokes_lid/cos2-k=2-nomass-fem2-512-lambdy=1e5/.result_state.json",
        "outputs/stokes_lid/cos2-k=3-mass-fem2-512-lambdy=1e5/.result_state.json",
        "outputs/stokes_lid/femonly-cos2-ref2-512/fem_vs_fns_state.json",
    ],
    "outputs/figures_and_tables/stokes_lid/result_fem&cos2-k=2,3.json",
)
mod.drawing_config["input_path"] = str(cos2_json)
mod.drawing_config["output_dir"] = "outputs/figures_and_tables/fem&cos2-k=2,3"
mod.main()

classic_json = merge_records(
    [
        "outputs/stokes_lid/classic-k=2-nomass-fem2-512-lambdy=1e3/.result_state.json",
        "outputs/stokes_lid/classic-k=3-mass-fem2-512-lambdy=1e3/.result_state.json",
        "outputs/stokes_lid/femonly-classic-ref2-512/fem_vs_fns_state.json",
    ],
    "outputs/figures_and_tables/stokes_lid/result_fem&classic-k=2,3.json",
)
mod.drawing_config["input_path"] = str(classic_json)
mod.drawing_config["output_dir"] = "outputs/figures_and_tables/fem&classic-k=2,3"
mod.main()
PY
run_py "$tmp_py"
rm -f "$tmp_py"
