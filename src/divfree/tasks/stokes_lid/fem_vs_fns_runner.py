"""FEM vs FNS comparison runner for 2D lid-driven cavity Stokes.

This script compares FEM (P2/P1) and FNS under similar global DOF budgets,
using a high-accuracy FEM reference (P3/P2, N=512) as the error baseline.
"""

from __future__ import annotations

import argparse
import sys
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running as a plain script from arbitrary working directories:
#   python stokes_lid/run_fem-vs-fns.py --config ...

import numpy as np

from divfree.common.quadrature import QuadratureConfig
from divfree.common.utils import Timer
from divfree.assembly.boundary import boundary_quadrature
from divfree.tasks.stokes_homogeneous.runner import _rate_table

from .checkpointing import CheckpointManager
from .fem_reference import ExactFEMEvaluator, FEMReferenceSolver2D
from .fns_solver import FNSSolver2D
from .lid_bc import build_lid_bc
from .metrics import MetricsEvaluator2D
from .visualization import Visualizer2D


def _quad_cfg(cfg: Dict[str, Any], key: str, fallback: str) -> QuadratureConfig:
    """Build quadrature config with fallback keys."""
    return QuadratureConfig(**cfg.get(key, cfg.get(fallback, {})))


def load_config(path: Path) -> Dict[str, Any]:
    """Load comparison JSON config."""
    cfg = json.loads(path.read_text())
    cfg.setdefault("experiment_name", "stokes_lid_fem_vs_fns")
    if "output_dir" not in cfg:
        cfg["output_dir"] = cfg.get("out_dir", "runs_lid")
    cfg["out_dir"] = cfg["output_dir"]
    cfg.setdefault("fem_compare", {})
    cfg.setdefault("fns", {})
    return cfg


def fem_total_dof_p2p1(mesh_n: int) -> int:
    """Total mixed DOFs for 2D P2/P1 on structured N x N grid: 9N^2 + 10N + 3."""
    n = int(mesh_n)
    return int(9 * n * n + 10 * n + 3)


def evaluate_fem_against_reference(
    fem_ckpt: Path,
    ref_ckpt: Path,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Compute FEM-vs-reference errors using exact evaluators at quadrature points."""
    ckpt = CheckpointManager()
    fem_obj = ckpt.load(fem_ckpt)
    ref_obj = ckpt.load(ref_ckpt)

    fem_eval = ExactFEMEvaluator(
        dofs_array=fem_obj["arrays"]["dofs_array"],
        mesh_N=int(np.asarray(fem_obj["arrays"]["mesh_N"]).ravel()[0]),
        fem_order=int(np.asarray(fem_obj["arrays"]["fem_order"]).ravel()[0]),
    )
    ref_eval = ExactFEMEvaluator(
        dofs_array=ref_obj["arrays"]["dofs_array"],
        mesh_N=int(np.asarray(ref_obj["arrays"]["mesh_N"]).ravel()[0]),
        fem_order=int(np.asarray(ref_obj["arrays"]["fem_order"]).ravel()[0]),
    )

    fns_cfg = config["fns"]
    quad = _quad_cfg(fns_cfg, key="evaluation", fallback="gauss_quad")
    bquad = _quad_cfg(fns_cfg.get("boundary", {}), key="evaluation", fallback="boundary_quad")

    err_l2 = 0.0
    err_h1 = 0.0
    div_fem2 = 0.0
    div_ref2 = 0.0

    for pts, wts in quad.iter(d=2):
        uf, gf = fem_eval.evaluate_at_points(pts)
        ur, gr = ref_eval.evaluate_at_points(pts)
        du = uf - ur
        dg = gf - gr
        err_l2 += float(np.sum(wts * np.sum(du * du, axis=1)))
        err_h1 += float(np.sum(wts * np.sum(dg * dg, axis=(1, 2))))
        div_fem2 += float(np.sum(wts * (gf[:, 0, 0] + gf[:, 1, 1]) ** 2))
        div_ref2 += float(np.sum(wts * (gr[:, 0, 0] + gr[:, 1, 1]) ** 2))

    lid = build_lid_bc(config.get("lid", {}))
    err_bdry = 0.0
    for pts, wts in boundary_quadrature(d=2, quad=bquad):
        uf, _ = fem_eval.evaluate_at_points(pts)
        gb = lid.eval_on_boundary(pts)
        err_bdry += float(np.sum(wts * np.sum((uf - gb) ** 2, axis=1)))

    return {
        "Err_L2": float(np.sqrt(err_l2)),
        "Err_H1": float(np.sqrt(err_h1)),
        "Err_bdry": float(np.sqrt(err_bdry)),
        "Div_fns": float(np.sqrt(div_fem2)),
        "Div_ref": float(np.sqrt(div_ref2)),
    }


def attach_rates_by_dof(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach log-slope rates using DOF as horizontal scale."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        grouped.setdefault(str(r["solver"]), []).append(r)

    out: List[Dict[str, Any]] = []
    for solver, rr in grouped.items():
        # Reuse existing rate helper by mapping DOF -> M.
        table = _rate_table(
            [{"M": int(r["DOF"]), "h1_err_rel": r["Err_H1"], "l2_err_rel": r["Err_L2"]} for r in rr]
        ).rename(columns={"h1_rate": "rate_H1", "l2_rate": "rate_L2"})
        m2row = {int(row.M): row for _, row in table.iterrows()}

        rr_sorted = sorted(rr, key=lambda x: int(x["DOF"]))
        rate_bdry = [math.nan]
        for i in range(1, len(rr_sorted)):
            e0, e1 = rr_sorted[i - 1]["Err_bdry"], rr_sorted[i]["Err_bdry"]
            d0, d1 = rr_sorted[i - 1]["DOF"], rr_sorted[i]["DOF"]
            if e0 > 0 and e1 > 0 and d0 > 0 and d1 > 0 and d0 != d1:
                rate_bdry.append(float(-math.log(e1 / e0) / math.log(d1 / d0)))
            else:
                rate_bdry.append(math.nan)

        for rb, rec in zip(rate_bdry, rr_sorted):
            row = m2row[int(rec["DOF"])]
            rec["rate_L2"] = float(row.rate_L2) if np.isfinite(row.rate_L2) else math.nan
            rec["rate_H1"] = float(row.rate_H1) if np.isfinite(row.rate_H1) else math.nan
            rec["rate_bdry"] = rb
            out.append(rec)

    return sorted(out, key=lambda z: (z["solver"], int(z["DOF"])))


def _print_record(rec: Dict[str, Any]) -> None:
    """Standardized terminal output for each computed record."""
    print(
        "[fem-vs-fns] "
        f"solver={rec['solver']} DOF={rec['DOF']} "
        f"Err_L2={rec['Err_L2']:.4e} Err_H1={rec['Err_H1']:.4e} "
        f"Err_bdry={rec['Err_bdry']:.4e} Div={rec['Div_fns']:.4e}"
    )


def _log(msg: str) -> None:
    print(f"[fem-vs-fns] {msg}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    run_dir = Path(cfg["output_dir"]) / str(cfg["experiment_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg_path, run_dir / "config.used.json")
    cfg["output_dir"] = str(run_dir)
    _log("=" * 72)
    _log(f"run_dir={run_dir}")
    _log(f"config={cfg_path}")
    _log("=" * 72)

    state_path = run_dir / "fem_vs_fns_state.json"
    records: List[Dict[str, Any]] = []
    if state_path.exists() and not bool(cfg.get("overwrite", False)):
        records = list(json.loads(state_path.read_text()).get("records", []))

    ckpt = CheckpointManager()
    fem_solver = FEMReferenceSolver2D(ckpt)
    fns_solver = FNSSolver2D(ckpt)
    metrics = MetricsEvaluator2D(ckpt)
    vis = Visualizer2D(ckpt)

    _log("Computing/loading high-accuracy FEM reference")
    high_ref_cfg = dict(cfg)
    high_ref_ref = dict(cfg.get("reference", {}))
    high_ref_ref.update(
        {
            "fem_mesh_N": int(cfg.get("fem_compare", {}).get("reference_mesh_N", 512)),
            "fem_order": int(cfg.get("fem_compare", {}).get("reference_fem_order", 2)),
            "checkpoint_path": str(run_dir / "reference" / "reference_exact.npz"),
            "reuse_if_exists": True,
        }
    )
    high_ref_cfg["reference"] = high_ref_ref
    with Timer() as t_ref:
        reference_ckpt = fem_solver.solve(high_ref_cfg)
    _log(f"reference={reference_ckpt} ({t_ref.dt:.2f}s)")

    done_keys = {(str(r.get("solver")), int(r.get("DOF", -1)), int(r.get("scale", -1))) for r in records}

    # FEM comparison loop (P2/P1).
    for mesh_n in cfg.get("fem_compare", {}).get("mesh_list", []):
        mesh_n = int(mesh_n)
        dof = fem_total_dof_p2p1(mesh_n)
        key = ("fem_p2p1", dof, mesh_n)
        if key in done_keys and not bool(cfg.get("overwrite", False)):
            _log(f"skip FEM mesh_N={mesh_n} (exists)")
            continue

        fem_cfg = dict(cfg)
        fem_ref = dict(cfg.get("reference", {}))
        fem_ref.update(
            {
                "fem_mesh_N": mesh_n,
                "fem_order": 2,
                "checkpoint_path": str(run_dir / "fem_compare" / f"fem_N{mesh_n}_p2p1.npz"),
                "reuse_if_exists": True,
            }
        )
        fem_cfg["reference"] = fem_ref

        with Timer() as t_fem:
            fem_ckpt = fem_solver.solve(fem_cfg)
        with Timer() as t_eval:
            m = evaluate_fem_against_reference(fem_ckpt, reference_ckpt, cfg)

        rec = {
            "solver": "fem_p2p1",
            "k": None,
            "scale": mesh_n,
            "DOF": dof,
            **m,
            "time_solve_sec": float(t_fem.dt),
            "time_eval_sec": float(t_eval.dt),
            "ckpt": str(fem_ckpt),
            # for Visualizer2D compatibility (x-axis field name)
            "M": dof,
        }
        records = [r for r in records if not (str(r.get("solver")) == "fem_p2p1" and int(r.get("scale", -1)) == mesh_n)]
        records.append(rec)
        done_keys.add(key)
        _print_record(rec)
        state_path.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False))

    # FNS comparison loop.
    solvers = cfg.get("fns", {}).get("solver", ["mass"])
    if isinstance(solvers, str):
        solvers = [solvers]

    for solver_name in solvers:
        for m_init in cfg.get("fns", {}).get("n_list", []):
            m_init = int(m_init)
            run_cfg = dict(cfg)
            run_cfg["_current_solver"] = solver_name
            run_cfg["_current_n"] = m_init

            with Timer() as t_fns:
                fns_ckpt = fns_solver.solve(run_cfg, reference=reference_ckpt)
            fns_obj = ckpt.load(fns_ckpt)
            dof = int(fns_obj["meta"].get("M", fns_obj["arrays"]["W"].shape[0]))
            key = (solver_name, dof, m_init)
            if key in done_keys and not bool(cfg.get("overwrite", False)):
                _log(f"skip FNS solver={solver_name} M_init={m_init} (exists)")
                continue

            with Timer() as t_eval:
                m = metrics.evaluate(fns_ckpt, reference_ckpt, cfg)

            rec = {
                "solver": solver_name,
                "k": int(cfg.get("fns", {}).get("reluk_k", 3)),
                "scale": m_init,
                "DOF": dof,
                **m,
                "time_solve_sec": float(t_fns.dt),
                "time_eval_sec": float(t_eval.dt),
                "ckpt": str(fns_ckpt),
                "M": dof,
            }
            records = [
                r
                for r in records
                if not (str(r.get("solver")) == solver_name and int(r.get("scale", -1)) == m_init)
            ]
            records.append(rec)
            done_keys.add(key)
            _print_record(rec)
            state_path.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False))

    if not records:
        raise RuntimeError("No records computed.")

    records = attach_rates_by_dof(records)

    csv_path = run_dir / "fem_vs_fns_metrics.csv"
    fieldnames = [
        "solver",
        "k",
        "DOF",
        "scale",
        "Err_L2",
        "Err_H1",
        "Err_bdry",
        "Div_fns",
        "Div_ref",
        "rate_L2",
        "rate_H1",
        "rate_bdry",
        "time_solve_sec",
        "time_eval_sec",
        "ckpt",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(records, key=lambda x: (x["solver"], int(x["DOF"]))):
            w.writerow({k: r.get(k, "") for k in fieldnames})

    _log(f"saved metrics: {csv_path}")
    vis.plot_convergence(records, cfg)
    _log("saved convergence figure under figs/")
    _log("=" * 72)


if __name__ == "__main__":
    main()
