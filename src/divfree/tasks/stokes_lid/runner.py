"""Main CLI for 2D Stokes lid-driven cavity experiments."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from divfree.common.utils import Timer


from .checkpointing import CheckpointManager
from .fem_reference import FEMReferenceSolver2D
from .fns_solver import FNSSolver2D
from .metrics import MetricsEvaluator2D
from .visualization import Visualizer2D


def load_config(path: Path) -> Dict[str, Any]:
    """Load experiment config directly from the user JSON file."""
    cfg = json.loads(path.read_text())
    if "output_dir" not in cfg:
        cfg["output_dir"] = cfg.get("out_dir", "runs_lid")
    cfg["out_dir"] = cfg["output_dir"]
    cfg.setdefault("experiment_name", "stokes_lid")
    cfg.setdefault("nu", float(cfg.get("problem", {}).get("nu", cfg.get("nu", 1.0))))
    cfg.setdefault("fns", {})
    cfg["fns"].setdefault("inner_evaluate", {"delta": 0.1})
    return cfg


def _log(msg: str) -> None:
    print(f"[stokes_lid] {msg}")


def _to_yaml(obj: Any, indent: int = 0) -> str:
    """Serialize a Python object to a simple YAML string (no external deps)."""
    sp = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{k}:")
                lines.append(_to_yaml(v, indent + 1))
            else:
                lines.append(f"{sp}{k}: {_yaml_scalar(v)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        lines = []
        for v in obj:
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(_to_yaml(v, indent + 1))
            else:
                lines.append(f"{sp}- {_yaml_scalar(v)}")
        return "\n".join(lines)
    return f"{sp}{_yaml_scalar(obj)}"


def _yaml_scalar(v: Any) -> str:
    """Render scalar value in YAML-friendly form."""
    import math

    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return ".nan"
        if math.isinf(v):
            return ".inf" if v > 0 else "-.inf"
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    return json.dumps(v, ensure_ascii=False)


def _save_result_yaml(path: Path, obj: Dict[str, Any]) -> None:
    """Save human-readable result.yaml without third-party yaml dependency."""
    path.write_text(_to_yaml(obj) + "\n", encoding="utf-8")


def _load_existing_results(state_path: Path) -> Tuple[List[Dict[str, Any]], set[Tuple[str, int]]]:
    """Load resumable records from internal JSON state file."""
    if not state_path.exists():
        return [], set()
    data = json.loads(state_path.read_text())
    rows = list(data.get("records", []))
    done = {(str(r["solver"]), int(r["M_init"])) for r in rows if "solver" in r and "M_init" in r}
    return rows, done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    run_dir = Path(cfg["output_dir"]) / str(cfg["experiment_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(run_dir)

    # Keep an exact copy of the command-line config JSON for reproducibility.
    shutil.copyfile(cfg_path, run_dir / "config.used.json")

    _log("=" * 72)
    _log(f"run_dir={run_dir}")
    _log(f"config={cfg_path}")
    _log("=" * 72)

    ckpt = CheckpointManager()
    fem_solver, fns_solver = FEMReferenceSolver2D(ckpt), FNSSolver2D(ckpt)
    evaluator, vis = MetricsEvaluator2D(ckpt), Visualizer2D(ckpt)

    result_path = run_dir / "result.yaml"
    state_path = run_dir / ".result_state.json"
    records, done_pairs = _load_existing_results(state_path)
    if done_pairs:
        _log(f"resume: found {len(done_pairs)} finished (solver, M_init) records; skipping existing.")

    _log("step=FEM reference")
    with Timer() as t_fem:
        fem_ckpt = fem_solver.solve(cfg)
    fem_time = float(t_fem.dt)
    _log(f"FEM done in {fem_time:.3f}s -> {fem_ckpt}")

    solvers = cfg["fns"].get("solver", "mass")
    if isinstance(solvers, str):
        solvers = [solvers]

    for solver_name in solvers:
        for M_init in map(int, cfg["fns"].get("n_list", [])):
            pair = (solver_name, M_init)
            if pair in done_pairs and not bool(cfg.get("overwrite", False)):
                _log(f"skip existing: solver={solver_name} M_init={M_init}")
                continue

            _log("-" * 72)
            _log(f"start solver={solver_name} M_init={M_init}")
            run_cfg = dict(cfg)
            run_cfg["_current_solver"] = solver_name
            run_cfg["_current_n"] = M_init

            with Timer() as t_fns:
                fns_ckpt = fns_solver.solve(run_cfg, reference=fem_ckpt)
            fns_obj = ckpt.load(fns_ckpt)
            M_eff = int(fns_obj["meta"].get("M", fns_obj["arrays"]["W"].shape[0]))
            _log(f"solve done in {float(t_fns.dt):.3f}s -> M={M_eff}")

            _log("evaluate metrics")
            with Timer() as t_eval:
                metrics = evaluator.evaluate(fns_ckpt, fem_ckpt, cfg)

            rec = {
                "solver": solver_name,
                "M_init": M_init,
                "M": M_eff,
                "k": int(cfg["fns"].get("reluk_k", 3)),
                "fns_ckpt": str(fns_ckpt),
                **metrics,
                "time_fem_sec": fem_time,
                "time_fns_total_sec": float(t_fns.dt),
                "time_eval_sec": float(t_eval.dt),
                "time_fns_assemble_sec": float(fns_obj["meta"].get("timing_sec", {}).get("assemble", float("nan"))),
                "time_fns_solve_sec": float(fns_obj["meta"].get("timing_sec", {}).get("solve", float("nan"))),
                "residual_l2": float(fns_obj["meta"].get("residual_l2", float("nan"))),
                "svd_min": fns_obj["meta"].get("svd_min", None),
                "svd_max": fns_obj["meta"].get("svd_max", None),
            }

            msg = (
                f"[stokes_lid] done solver={solver_name} M_init={M_init} M={M_eff} "
                f"L2={rec['Err_L2']:.3e} H1={rec['Err_H1']:.3e} Bdry={rec['Err_bdry']:.3e}"
            )
            if "Err_L2_inner" in rec:
                msg += f" L2_inner={rec['Err_L2_inner']:.3e} H1_inner={rec['Err_H1_inner']:.3e}"
            _log(msg.replace("[stokes_lid] ", ""))

            records = [r for r in records if not (str(r.get("solver")) == solver_name and int(r.get("M_init", -1)) == M_init)]
            records.append(rec)
            done_pairs.add(pair)
            state_path.write_text(json.dumps({"fem_ckpt": str(fem_ckpt), "records": records}, ensure_ascii=False, indent=2))
            _save_result_yaml(result_path, {"fem_ckpt": str(fem_ckpt), "records": records})

    if not records:
        raise RuntimeError("No records found. Check fns.solver / fns.n_list settings.")

    records = evaluator.attach_rates(records)

    _log("=" * 72)
    _log("rates summary")
    for rec in sorted(records, key=lambda r: (r["solver"], int(r["M"]))):
        msg = (
            f"  solver={rec['solver']} M={rec['M']} "
            f"rate_L2={rec.get('rate_L2', float('nan'))} "
            f"rate_H1={rec.get('rate_H1', float('nan'))} "
            f"rate_bdry={rec.get('rate_bdry', float('nan'))}"
        )
        if "rate_L2_inner" in rec:
            msg += f" rate_L2_inner={rec.get('rate_L2_inner')} rate_H1_inner={rec.get('rate_H1_inner')}"
        _log(msg.strip())

    metrics_csv = run_dir / "metrics.csv"
    field_order = [
        "solver", "k", "M", "M_init",
        "Err_L2", "Err_H1", "Err_bdry", "Err_L2_inner", "Err_H1_inner",
        "rate_L2", "rate_H1", "rate_bdry", "rate_L2_inner", "rate_H1_inner",
    ]
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r["solver"], int(r["M"]))):
            writer.writerow({k: rec.get(k, "") for k in field_order})
    _log(f"saved metrics: {metrics_csv}")

    state_path.write_text(json.dumps({"fem_ckpt": str(fem_ckpt), "records": records}, ensure_ascii=False, indent=2))
    _save_result_yaml(result_path, {"fem_ckpt": str(fem_ckpt), "records": records})
    _log(f"saved detailed results: {result_path}")

    if cfg.get("plotting", {}).get("make_flow_figure", True):
        best = max(records, key=lambda r: int(r["M"]))
        _log(f"plot flow with solver={best['solver']} M={best['M']}")
        vis.plot_flow(fem_ckpt, best["fns_ckpt"], cfg)

    if cfg.get("plotting", {}).get("make_convergence_figure", True):
        _log("plot convergence")
        vis.plot_convergence(records, cfg)
    _log("=" * 72)


if __name__ == "__main__":
    main()
