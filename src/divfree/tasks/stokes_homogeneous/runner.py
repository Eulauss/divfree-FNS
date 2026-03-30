from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from divfree.common.io_utils import save_config, save_record, save_matrix, save_parameters
from divfree.common.quadrature import QuadratureConfig
from divfree.common.sampling import NeuronSampler
from divfree.common.solvers import build_solver
from divfree.common.utils import Timer, ensure_dir, now_ts, set_seed, write_json

from divfree.tasks.l2_approx.targets import TargetBase

from divfree.assembly.stokes_mass import assemble_stokes_2d, assemble_stokes_3d
from divfree.assembly.boundary import BoundaryConfig, boundary_quadrature
from .targets import build_target
from divfree.assembly.stokes_nomass import assemble_stokes_2d_nomass, assemble_stokes_3d_nomass


def _predict_divfree_2d(pts: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    """
    approximator of div-free 2d vector fields, coming from one potential

    pts: (B,2)  B: batch size, evluate B points at the same time
    W: (M,2)        b:(M,1)     a(M,1)
    """
    z = pts @ W.T + b[None, :]
    # s = d/dt ReLU(t)^k
    if int(k) == 1:
        S = (z > 0).astype(np.float64)
    else:
        relu = np.maximum(z, 0.0)
        S = float(k) * np.power(relu, int(k) - 1)
    w1 = W[:, 0]
    w2 = W[:, 1]
    t1 = a * w2
    t2 = -a * w1
    v1 = S @ t1
    v2 = S @ t2
    return np.stack([v1, v2], axis=1)


def _predict_divfree_2d_grad(pts: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    """Return grad u with shape (n,2,2)."""
    if int(k) < 2:
        raise ValueError("k must be >= 2 to evaluate H1 seminorm")
    z = pts @ W.T + b[None, :]
    if int(k) == 2:
        S2 = 2.0 * (z > 0).astype(np.float64)
    else:
        relu = np.maximum(z, 0.0)
        S2 = float(k * (k - 1)) * np.power(relu, int(k) - 2)

    w1 = W[:, 0]
    w2 = W[:, 1]
    coeff = S2 * a[None, :]

    t1 = coeff * w2[None, :]
    t2 = coeff * (-w1[None, :])

    grad = np.zeros((pts.shape[0], 2, 2), dtype=np.float64)
    grad[:, 0, 0] = t1 @ w1
    grad[:, 0, 1] = t1 @ w2
    grad[:, 1, 0] = t2 @ w1
    grad[:, 1, 1] = t2 @ w2
    return grad


def _predict_divfree_3d(pts: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    """
    approxiamator of div-free 3d vector fields, coming from 3 potentials (sharing the same inner-coefficients)
    resembled by anti-symmetric matrix

    pts: (B,3)  B: batch size, evluate B points at the same time
    W: (M,3)        b:(3M,1)     a(3M,1)
    """
    M = W.shape[0]
    a12 = a[:M]
    a13 = a[M:2 * M]
    a23 = a[2 * M:3 * M]
    # s = d/dt ReLU(t)^k
    z = pts @ W.T + b[None, :]
    if int(k) == 1:
        S = (z > 0).astype(np.float64)
    else:
        relu = np.maximum(z, 0.0)
        S = float(k) * np.power(relu, int(k) - 1)
    # resemble by anti-symmetric vector fields
    w1 = W[:, 0]
    w2 = W[:, 1]
    w3 = W[:, 2]
    t1 = a12 * w2 + a13 * w3
    t2 = -a12 * w1 + a23 * w3
    t3 = -a13 * w1 - a23 * w2
    v1 = S @ t1
    v2 = S @ t2
    v3 = S @ t3
    return np.stack([v1, v2, v3], axis=1)


def _predict_divfree_3d_grad(pts: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    """Return grad u with shape (n,3,3)."""
    if int(k) < 2:
        raise ValueError("k must be >= 2 to evaluate H1 seminorm")
    M = W.shape[0]
    a12 = a[:M]
    a13 = a[M:2 * M]
    a23 = a[2 * M:3 * M]

    z = pts @ W.T + b[None, :]
    if int(k) == 2:
        S2 = 2.0 * (z > 0).astype(np.float64)
    else:
        relu = np.maximum(z, 0.0)
        S2 = float(k * (k - 1)) * np.power(relu, int(k) - 2)

    w1 = W[:, 0]
    w2 = W[:, 1]
    w3 = W[:, 2]

    c1 = a12 * w2 + a13 * w3
    c2 = -a12 * w1 + a23 * w3
    c3 = -a13 * w1 - a23 * w2

    coeff = S2
    t1 = coeff * c1[None, :]
    t2 = coeff * c2[None, :]
    t3 = coeff * c3[None, :]

    grad = np.zeros((pts.shape[0], 3, 3), dtype=np.float64)
    grad[:, 0, 0] = t1 @ w1
    grad[:, 0, 1] = t1 @ w2
    grad[:, 0, 2] = t1 @ w3
    grad[:, 1, 0] = t2 @ w1
    grad[:, 1, 1] = t2 @ w2
    grad[:, 1, 2] = t2 @ w3
    grad[:, 2, 0] = t3 @ w1
    grad[:, 2, 1] = t3 @ w2
    grad[:, 2, 2] = t3 @ w3
    return grad


def _h1_error(
    problem_type: str,
    target: TargetBase,
    W: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    k: int,
    eval_quad: QuadratureConfig,
    boundary_cfg: BoundaryConfig,
):
    """
    Compute H1 seminorm error, interior L2 error, and boundary L2 error.

    return:
        h1_err_abs: absolute H1 seminorm error
        h1_err_rel: relative H1 seminorm error
        target_h1: target H1 seminorm
        boundary_l2: L2 norm on boundary of u_M
        l2_err_abs: L2 error in domain
        l2_err_rel: relative L2 error in domain
        target_l2: L2 norm of target in domain
    """
    d = {"stokes2d": 2, "stokes3d": 3}[problem_type]
    err2 = 0.0
    norm2 = 0.0
    l2_err2 = 0.0
    l2_norm2 = 0.0

    for pts, wts in eval_quad.iter(d=d):
        grad_true = target.eval_grad(pts)
        u_true = target.eval(pts)
        if problem_type == "stokes2d":
            grad_pred = _predict_divfree_2d_grad(pts, W, b, a, k)
            u_pred = _predict_divfree_2d(pts, W, b, a, k)
        elif problem_type == "stokes3d":
            grad_pred = _predict_divfree_3d_grad(pts, W, b, a, k)
            u_pred = _predict_divfree_3d(pts, W, b, a, k)
        else:
            raise ValueError(problem_type)

        diff = grad_true - grad_pred
        diff2 = np.sum(diff * diff, axis=(1, 2))
        err2 += float(np.sum(wts * diff2))
        norm2 += float(np.sum(wts * np.sum(grad_true * grad_true, axis=(1, 2))))

        diff_u = u_true - u_pred
        diff_u2 = np.sum(diff_u * diff_u, axis=1)
        l2_err2 += float(np.sum(wts * diff_u2))
        l2_norm2 += float(np.sum(wts * np.sum(u_true * u_true, axis=1)))

    # boundary L2 error of u_M
    boundary_err = 0.0
    for boundary_pts, boundary_wts in boundary_quadrature(d=d, quad=boundary_cfg.quad):
        if problem_type == "stokes2d":
            u_hat = _predict_divfree_2d(boundary_pts, W, b, a, k)
        else:
            u_hat = _predict_divfree_3d(boundary_pts, W, b, a, k)
        boundary_err += float(np.sum(boundary_wts * np.sum(u_hat * u_hat, axis=1)))

    rel = np.sqrt(err2 / max(norm2, 1e-30))
    absn = np.sqrt(err2)
    return {
        "h1_err_abs": float(absn),
        "h1_err_rel": float(rel),
        "target_h1": float(np.sqrt(norm2)),
        "boundary_l2": float(np.sqrt(boundary_err)),
        "l2_err_abs": float(np.sqrt(l2_err2)),
        "l2_err_rel": float(np.sqrt(l2_err2 / max(l2_norm2, 1e-30))),
        "target_l2": float(np.sqrt(l2_norm2)),
    }


def _rate_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    draw a dataframe of approximation rate computation
    """
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values("M").drop_duplicates(subset=["M"], keep="last").reset_index(drop=True)
    Ms = df["M"].to_numpy()
    h1_errs = df["h1_err_rel"].to_numpy()
    l2_errs = df["l2_err_rel"].to_numpy()
    h1_rates = [np.nan]
    l2_rates = [np.nan]
    for i in range(len(df) - 1):
        m0, m1 = float(Ms[i]), float(Ms[i + 1])
        h0, h1 = float(h1_errs[i]), float(h1_errs[i + 1])
        l0, l1 = float(l2_errs[i]), float(l2_errs[i + 1])
        if h0 <= 0 or h1 <= 0 or m0 <= 0 or m1 <= 0:
            h1_rates.append(np.nan)
        else:
            h1_rates.append(-np.log(h1 / h0) / np.log(m1 / m0))
        if l0 <= 0 or l1 <= 0 or m0 <= 0 or m1 <= 0:
            l2_rates.append(np.nan)
        else:
            l2_rates.append(-np.log(l1 / l0) / np.log(m1 / m0))
    df["rate"] = h1_rates
    df["h1_rate"] = h1_rates
    df["l2_rate"] = l2_rates
    return df


def _augment_tikhonov(A: np.ndarray, b: np.ndarray, reg_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    if not reg_lambda or reg_lambda <= 0:
        return A, b
    M = A.shape[1]
    sqrt_lambda = float(np.sqrt(reg_lambda))
    A_aug = np.vstack([A, sqrt_lambda * np.eye(M)])
    b_aug = np.concatenate([b, np.zeros(M, dtype=np.float64)])
    return A_aug, b_aug


@dataclass
class ExperimentRunner:
    cfg: Dict[str, Any]

    def run(self) -> Path:
        cfg = copy.deepcopy(self.cfg)

        run_cfg = cfg.get("run", {})
        out_dir = Path(run_cfg.get("out_dir", cfg.get("out_dir", "runs")))
        name = run_cfg.get("name", f"run-{now_ts()}")
        seed = int(run_cfg.get("seed", 0))

        run_dir = ensure_dir(out_dir / name)
        set_seed(seed)
        save_config(run_dir, cfg)

        problem = cfg.get("problem", {})
        problem_type = str(problem.get("type", "stokes2d")).lower()
        target_cfg = problem.get("target", {})
        nu = float(problem.get("nu", 1.0))

        basis_cfg = cfg.get("basis", {})
        k = int(basis_cfg.get("k", 2))
        if k < 2:
            raise ValueError("k must be >= 2 for Stokes H1 seminorm training")

        sampler_cfg = cfg.get("sampler", {})
        d = {"stokes2d": 2, "stokes3d": 3}[problem_type]
        sampler = NeuronSampler(d=d, **sampler_cfg)

        quad_cfg = QuadratureConfig(**cfg.get("quadrature", {}))
        eval_cfg = QuadratureConfig(**cfg.get("evaluation", cfg.get("quadrature", {})))

        exp_cfg = cfg.get("experiment", {})
        M_list = list(exp_cfg.get("M_list", [64, 128, 256, 512, 1024, 2048]))
        reg_lambda = float(exp_cfg.get("reg_lambda", 1e-10))
        use_mass_matrix = bool(exp_cfg.get("mass_matrix", True))
        io_cfg = cfg.get("io", {}) or {}
        save_matrices = bool(io_cfg.get("save_matrices", True))
        save_parameters_flag = bool(io_cfg.get("save_parameters", False))

        boundary_cfg_raw = cfg.get("boundary", {})
        boundary_quad = QuadratureConfig(
            Nx=int(boundary_cfg_raw.get("Nx", boundary_cfg_raw.get("nx", 8))),
            order=int(boundary_cfg_raw.get("order", 3)),
            chunk_size=int(boundary_cfg_raw.get("chunk_size", 6000)),
        )
        boundary_cfg = BoundaryConfig(
            lambda_boundary=float(boundary_cfg_raw.get("lambda_boundary", boundary_cfg_raw.get("lambda", 1.0))),
            quad=boundary_quad,
        )

        solver = build_solver(cfg.get("solver", {"name": "auto"}), reg_lambda=reg_lambda if not use_mass_matrix else None)
        if not use_mass_matrix and solver.name not in {"lstsq", "lsqr", "qr"}:
            raise ValueError("mass_matrix=False requires solver 'lstsq', 'lsqr', or 'qr'")

        target = build_target(problem_type, target_cfg)

        all_records: List[Dict[str, Any]] = []
        results_path = run_dir / "results.jsonl"
        done_M = set()
        if results_path.exists():
            for line in results_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec0 = json.loads(line)
                    if "M_init" in rec0:
                        done_M.add(int(rec0["M_init"]))
                        all_records.append(rec0)
                except Exception:
                    continue

        if results_path.exists():
            if done_M:
                preview = sorted(done_M)
                preview_str = ", ".join(str(x) for x in preview[:12])
                more = "" if len(preview) <= 12 else f" (and {len(preview) - 12} more)"
                print(f"Found existing results for M_init: {preview_str}{more}. Will skip these M values.")
            else:
                print("Warning: results.jsonl exists but no valid records were parsed; rerunning all M.")

        for M_init in M_list:
            M_init = int(M_init)
            if M_init in done_M:
                print(f"\n=== M_init={M_init} already exists in results.jsonl; skipping ===")
                continue

            print(f"\n=== M_init ={M_init} | problem={problem_type} | k={k} | solver={solver.name} ===")
            W, b = sampler.sample(M_init)
            print(f"After filering, number of neurons used: {W.shape[0]}")

            with Timer() as t_asm:
                if use_mass_matrix:
                    if problem_type == "stokes2d":
                        res = assemble_stokes_2d(W, b, target.eval_grad, k, quad_cfg, reg_lambda, boundary_cfg, nu)
                    elif problem_type == "stokes3d":
                        res = assemble_stokes_3d(W, b, target.eval_grad, k, quad_cfg, reg_lambda, boundary_cfg, nu)
                    else:
                        raise ValueError(problem_type)
                else:
                    if problem_type == "stokes2d":
                        res = assemble_stokes_2d_nomass(W, b, target.eval_grad, k, quad_cfg, boundary_cfg, nu)
                    elif problem_type == "stokes3d":
                        res = assemble_stokes_3d_nomass(W, b, target.eval_grad, k, quad_cfg, boundary_cfg, nu)
                    else:
                        raise ValueError(problem_type)
            asm_time = t_asm.dt

            if save_matrices:
                save_matrix(run_dir, name=f"A_b_M{M_init}", A=res.A, b=res.b)

            with Timer() as t_sol:
                A_solve = res.A
                b_solve = res.b
                if not use_mass_matrix and solver.name in {"lstsq", "qr"}:
                    A_solve, b_solve = _augment_tikhonov(A_solve, b_solve, reg_lambda)
                a, sol_info = solver.solve(A_solve, b_solve)
            sol_time = t_sol.dt

            print(f"sovle done in {sol_time:.3f}s. Start evaluation.")
            with Timer() as t_eval:
                err = _h1_error(problem_type, target, W, b, a, k, eval_cfg, boundary_cfg)
            eval_time = t_eval.dt

            if solver.name == "lstsq" and getattr(solver, "save_svd", False):
                singular_values = sol_info.pop("singular_values", None)
                if singular_values is not None:
                    svd_dir = ensure_dir(run_dir / "svd")
                    svd_path = svd_dir / f"singular_values{M_init}.json"
                    write_json(svd_path, singular_values)
                    print(f"Saved singular values computed by lstsq to {svd_path}.")

            if save_parameters_flag:
                save_parameters(
                    run_dir,
                    problem_name=problem_type,
                    M=int(W.shape[0]),
                    W=W,
                    b=b,
                    a=a,
                    k=int(k),
                    problem_type=problem_type,
                )

            rec = {
                "M_init": int(M_init),
                "M": int(W.shape[0]),
                "dof": tuple(int(x) for x in res.A.shape),
                "k": int(k),
                "nu": float(nu),
                "reg_lambda": float(reg_lambda),
                "mass_matrix": bool(use_mass_matrix),
                "problem": {
                    "type": problem_type,
                    "target": target_cfg,
                },
                "sampler": sampler_cfg,
                "lambda_boundary": float(boundary_cfg.lambda_boundary),
                "boundary": {
                    "Nx": boundary_cfg.quad.Nx,
                    "order": boundary_cfg.quad.order,
                    "chunk_size": boundary_cfg.quad.chunk_size,
                },
                "quadrature": {"Nx": quad_cfg.Nx, "order": quad_cfg.order, "chunk_size": quad_cfg.chunk_size},
                "evaluation": {"Nx": eval_cfg.Nx, "order": eval_cfg.order, "chunk_size": eval_cfg.chunk_size},
                "timing_sec": {"assemble": float(asm_time), "solve": float(sol_time), "eval": float(eval_time)},
                **err,
                "solver_info": sol_info,
            }

            save_record(run_dir, rec)
            all_records.append(rec)

            print(
                f"assemble={asm_time:.3f}s | solve={sol_time:.3f}s | eval={eval_time:.3f}s | rel_H1={rec['h1_err_rel']:.3e}"
            )

        df = _rate_table(all_records)
        df.to_csv(run_dir / "rate_table.csv", index=False)
        print(f"\nSaved rate table to: {run_dir / 'rate_table.csv'}")
        return run_dir
