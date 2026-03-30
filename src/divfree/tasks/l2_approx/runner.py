from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.linalg

from divfree.assembly.l2_mass import assemble_scalar, assemble_divfree_2d, assemble_divfree_3d
from divfree.common.io_utils import save_config, save_record, save_matrix, save_parameters
from divfree.common.quadrature import QuadratureConfig, volume_of_cube
from divfree.common.sampling import NeuronSampler
from divfree.common.solvers import build_solver
from .targets import build_target, TargetBase
from divfree.common.utils import Timer, ensure_dir, now_ts, set_seed, write_json
from divfree.assembly.l2_nomass import (
    assemble_scalar_nomass,
    assemble_divfree_2d_nomass,
    assemble_divfree_3d_nomass,
)


def _predict_scalar(pts: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    '''
    approximator of scalar valued function

    pts: (B,d)  B: batch size, evluate B points at the same time
    W: (M,d)        b:(M,1)     a(M,1)
    '''
    z = pts @ W.T + b[None, :]
    phi = np.maximum(z, 0.0)
    if k != 1:
        phi = np.power(phi, int(k))
    return phi @ a


def _predict_divfree_2d(pts: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    '''
    approxiamator of div-free 2d vector fields, coming from one potential

    pts: (B,2)  B: batch size, evluate B points at the same time
    W: (M,2)        b:(M,1)     a(M,1)
    '''
    z = pts @ W.T + b[None, :]
    # s = d/dt ReLU(t)^k
    if int(k) == 1:
        S = (z > 0).astype(np.float64)
    else:
        relu = np.maximum(z, 0.0)
        S = float(k) * np.power(relu, int(k) - 1)
    w1 = W[:, 0]; w2 = W[:, 1]
    t1 = a * w2
    t2 = -a * w1
    v1 = S @ t1
    v2 = S @ t2
    return np.stack([v1, v2], axis=1)


def _predict_divfree_3d(pts: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    '''
    approxiamator of div-free 3d vector fields, coming from 3 potentials (sharing the same inner-coefficients)
    resembled by anti-symmetric matrix

    pts: (B,3)  B: batch size, evluate B points at the same time
    W: (M,3)        b:(3M,1)     a(3M,1)
    '''
    M = W.shape[0]
    a12 = a[:M]
    a13 = a[M:2*M]
    a23 = a[2*M:3*M]
    # s = d/dt ReLU(t)^k
    z = pts @ W.T + b[None, :]
    if int(k) == 1:
        S = (z > 0).astype(np.float64)
    else:
        relu = np.maximum(z, 0.0)
        S = float(k) * np.power(relu, int(k) - 1)
    # resemble by anti-symmetric vector fields
    w1 = W[:, 0]; w2 = W[:, 1]; w3 = W[:, 2]
    t1 = a12 * w2 + a13 * w3
    t2 = -a12 * w1 + a23 * w3
    t3 = -a13 * w1 - a23 * w2
    v1 = S @ t1
    v2 = S @ t2
    v3 = S @ t3
    return np.stack([v1, v2, v3], axis=1)


def _l2_error(problem_type: str, target: TargetBase, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int, eval_quad: QuadratureConfig):
    '''
    compute the predicted error with target

    return: 
        l2_err_abs: absolute square error
        l2_err_rel: reletive square error, dividing l2 of target
        target_l2: l2 of target
    '''
    d = target.input_dim()
    err2 = 0.0
    norm2 = 0.0

    for pts, wts in eval_quad.iter(d=d):
        y = target.eval(pts)
        if problem_type in {"scalar1d", "scalar"}:
            yhat = _predict_scalar(pts, W, b, a, k)
            diff2 = (y.reshape(-1) - yhat) ** 2
            err2 += float(np.sum(wts * diff2))
            norm2 += float(np.sum(wts * (y.reshape(-1) ** 2)))
        elif problem_type == "divfree2d":
            yhat = _predict_divfree_2d(pts, W, b, a, k)
            diff = y - yhat
            diff2 = np.sum(diff * diff, axis=1)
            err2 += float(np.sum(wts * diff2))
            norm2 += float(np.sum(wts * np.sum(y*y, axis=1)))
        elif problem_type == "divfree3d":
            yhat = _predict_divfree_3d(pts, W, b, a, k)
            diff = y - yhat
            diff2 = np.sum(diff * diff, axis=1)
            err2 += float(np.sum(wts * diff2))
            norm2 += float(np.sum(wts * np.sum(y*y, axis=1)))
        else:
            raise ValueError(problem_type)

    rel = np.sqrt(err2 / max(norm2, 1e-30))
    absn = np.sqrt(err2)
    return {"l2_err_abs": absn, "l2_err_rel": float(rel), "target_l2": float(np.sqrt(norm2))}


def _rate_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    '''
    draw a dataframe of approximation rate computation
    '''
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values("M").drop_duplicates(subset=["M"], keep="last").reset_index(drop=True)
    # Finite Element style rate based on relative error by default
    errs = df["l2_err_rel"].to_numpy()
    Ms = df["M"].to_numpy()
    rates = [np.nan]
    for i in range(len(df) - 1):
        e0, e1 = float(errs[i]), float(errs[i+1])
        m0, m1 = float(Ms[i]), float(Ms[i+1])
        if e0 <= 0 or e1 <= 0 or m0 <= 0 or m1 <= 0:
            rates.append(np.nan)
        else:
            rates.append( - np.log(e1/e0) / np.log(m1/m0))
    df["rate"] = rates
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
        problem_type = str(problem.get("type", "divfree3d")).lower()
        target_cfg = problem.get("target", {})

        basis_cfg = cfg.get("basis", {})
        k = int(basis_cfg.get("k", 2))

        sampler_cfg = cfg.get("sampler", {})

        quad_cfg = QuadratureConfig(**cfg.get("quadrature", {}))
        eval_cfg = QuadratureConfig(**cfg.get("evaluation", cfg.get("quadrature", {})))

        exp_cfg = cfg.get("experiment", {})
        M_list = list(exp_cfg.get("M_list", [64, 128, 256, 512, 1024, 2048]))
        reg_lambda = float(exp_cfg.get("reg_lambda", 1e-10))
        use_mass_matrix = bool(exp_cfg.get("mass_matrix", True))
        compute_cond_num = bool(exp_cfg.get("compute_cond_num", False))

        io_cfg = cfg.get("io", {}) or {}
        save_matrices = bool(io_cfg.get("save_matrices", True))
        save_parameters_flag = bool(io_cfg.get("save_parameters", False))

        solver = build_solver(cfg.get("solver", {"name": "auto"}), reg_lambda=reg_lambda if not use_mass_matrix else None)
        if not use_mass_matrix and solver.name not in {"lstsq", "lsqr", "qr"}:
            raise ValueError("mass_matrix=False requires solver 'lstsq', 'lsqr', or 'qr'")

        target = build_target(problem_type, target_cfg)
        sampler = NeuronSampler(d=target.input_dim(), **sampler_cfg)

        all_records: List[Dict[str, Any]] = []
        # If results.jsonl exists, skip completed M values and include them in final rate table.
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
                more = "" if len(preview) <= 12 else f" (and {len(preview)-12} more)"
                print(f"Found existing results for M_init: {preview_str}{more}. Will skip these M values.")
            else:
                # File exists but nothing parsed; warn once.
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
                    if problem_type in {"scalar1d", "scalar"}:
                        res = assemble_scalar(W, b, target.eval, k, quad_cfg, reg_lambda=reg_lambda)
                    elif problem_type == "divfree2d":
                        res = assemble_divfree_2d(W, b, target.eval, k, quad_cfg, reg_lambda=reg_lambda)
                    elif problem_type == "divfree3d":
                        res = assemble_divfree_3d(W, b, target.eval, k, quad_cfg, reg_lambda=reg_lambda)
                    else:
                        raise ValueError(problem_type)
                else:
                    if problem_type in {"scalar1d", "scalar"}:
                        res = assemble_scalar_nomass(W, b, target.eval, k, quad_cfg)
                    elif problem_type == "divfree2d":
                        res = assemble_divfree_2d_nomass(W, b, target.eval, k, quad_cfg)
                    elif problem_type == "divfree3d":
                        res = assemble_divfree_3d_nomass(W, b, target.eval, k, quad_cfg)
                    else:
                        raise ValueError(problem_type)
            asm_time = t_asm.dt
            print(f"Assembly done in {asm_time:.3f}s.")

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
                err = _l2_error(problem_type, target, W, b, a, k, eval_cfg)
            eval_time = t_eval.dt

            if solver.name == "lstsq" and getattr(solver, "save_svd", False):
                singular_values = sol_info.pop("singular_values", None)
                if singular_values is not None:
                    svd_dir = ensure_dir(run_dir / "svd")
                    svd_path = svd_dir / f"singular_values{M_init}.json"
                    write_json(svd_path, singular_values)
                    print(f"Saved singular values computed by lstsq to {svd_path}.")

            if compute_cond_num:
                true_svd_dir = ensure_dir(run_dir / "singular_values_direct_compute")
                true_svd_path = true_svd_dir / f"singular_values{M_init}.json"
                
                print("Computing full SVD for validation...")
                
                # including spare matrix
                if scipy.sparse.issparse(A_solve):
                    A_target = A_solve.toarray()
                else:
                    A_target = np.asarray(A_solve)
                
                s_desc = scipy.linalg.svd(A_target, compute_uv=False)
                true_svd = s_desc[::-1]
                write_json(true_svd_path, true_svd.tolist())
                cond_num = true_svd[-1] / true_svd[0] if true_svd[0] > 0 else float('inf')
                print(f"Saved singular values. Condition Number: {cond_num:.4e}")
                print(f"Saved path: {true_svd_path}.")

            if save_parameters_flag:
                params_path = save_parameters(
                    run_dir,
                    problem_name=problem_type,
                    M=int(W.shape[0]),
                    W=W,
                    b=b,
                    a=a,
                    k=int(k),
                    problem_type=problem_type,
                )
                print(f"Saved parameters to {params_path}.")

            rec = {
                "M_init": int(M_init),
                "M": int(W.shape[0]),
                "dof": tuple(int(x) for x in res.A.shape),
                "problem_type": problem_type,
                "target": target.name,
                "k": int(k),
                "reg_lambda": float(reg_lambda),
                "mass_matrix": bool(use_mass_matrix),
                "problem": {
                    "type": problem_type,
                    "target": target_cfg,
                },
                "sampler": sampler_cfg,
                "quadrature": {"Nx": quad_cfg.Nx, "order": quad_cfg.order, "chunk_size": quad_cfg.chunk_size},
                "evaluation": {"Nx": eval_cfg.Nx, "order": eval_cfg.order, "chunk_size": eval_cfg.chunk_size},
                "timing_sec": {"assemble": float(asm_time), "solve": float(sol_time), "eval": float(eval_time)},
                **err,
                "solver_info": sol_info,
            }
            if compute_cond_num:
                rec["2-condition number computed by svd"] = float(cond_num)

            save_record(run_dir, rec)
            all_records.append(rec)

            print(f"assemble={asm_time:.3f}s | solve={sol_time:.3f}s | eval={eval_time:.3f}s | rel_L2={rec['l2_err_rel']:.3e}")

        df = _rate_table(all_records)
        df.to_csv(run_dir / "rate_table.csv", index=False)
        print(f"\nSaved rate table to: {run_dir / 'rate_table.csv'}")
        return run_dir
