"""Glue code from lid experiment config to existing FNS Stokes assemblers/solvers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from divfree.common.activation import relu_k_derivative_scale
from divfree.common.quadrature import QuadratureConfig
from divfree.common.sampling import NeuronSampler
from divfree.common.solvers import build_solver
from divfree.common.utils import Timer
from divfree.assembly.stokes_nomass import assemble_stokes_2d_nomass
from divfree.assembly.stokes_mass import assemble_stokes_2d
from divfree.assembly.boundary import BoundaryConfig, boundary_quadrature

from .checkpointing import CheckpointManager
from .lid_bc import build_lid_bc


def _quad_cfg(fns_cfg: Dict[str, Any], key: str, fallback: str) -> QuadratureConfig:
    """Build quadrature config with backward-compatible fallback keys."""
    return QuadratureConfig(**fns_cfg.get(key, fns_cfg.get(fallback, {})))


@dataclass
class FNSSolver2D:
    """Solve 2D lid-driven Stokes with FNS basis using existing repo assemblers."""

    ckpt: CheckpointManager

    def _boundary_rhs_mass(self, W: np.ndarray, bias: np.ndarray, k: int, boundary_cfg: BoundaryConfig, lid_cfg: Dict[str, Any]) -> np.ndarray:
        """Assemble +lambda<g,phi_i> boundary RHS for mass strategy."""
        w1, w2 = W[:, 0], W[:, 1]
        rhs = np.zeros((W.shape[0],), dtype=np.float64)
        bc = build_lid_bc(lid_cfg)
        lam = float(boundary_cfg.lambda_boundary)
        for pts, wts in boundary_quadrature(d=2, quad=boundary_cfg.quad):
            S = relu_k_derivative_scale(pts @ W.T + bias[None, :], k)
            g = bc.eval_on_boundary(pts)
            rhs += lam * np.sum(wts[:, None] * S * (g[:, 0:1] * w2[None, :] - g[:, 1:2] * w1[None, :]), axis=0)
        return rhs

    def _append_boundary_rows_nomass(
        self, A: np.ndarray, bvec: np.ndarray, W: np.ndarray, bias: np.ndarray, k: int, boundary_cfg: BoundaryConfig, lid_cfg: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Append weighted boundary rows for direct-LS strategy."""
        w1, w2 = W[:, 0], W[:, 1]
        bc = build_lid_bc(lid_cfg)
        lam_weight = np.sqrt(float(boundary_cfg.lambda_boundary))
        rowsA, rowsb = [A], [bvec]
        for pts, wts in boundary_quadrature(d=2, quad=boundary_cfg.quad):
            S = relu_k_derivative_scale(pts @ W.T + bias[None, :], k)
            sqrtw = np.sqrt(wts)[:, None]
            base = S * (sqrtw * lam_weight)
            g = bc.eval_on_boundary(pts)
            rowsA.extend([base * w2[None, :], base * (-w1[None, :])])
            rowsb.extend([sqrtw[:, 0] * lam_weight * g[:, 0], sqrtw[:, 0] * lam_weight * g[:, 1]])
        return np.vstack(rowsA), np.concatenate(rowsb)

    def solve(self, config: Dict[str, Any], reference: Any = None) -> Path:
        """Run one FNS solve for configured M_init and solver kind, then save checkpoint."""
        out_dir = Path(config["output_dir"])
        fns_cfg = config["fns"]
        M_init = int(config["_current_n"])
        solver_kind = str(config["_current_solver"])

        ckpt_path = out_dir / "fns" / f"{solver_kind}_Minit{M_init}.npz"
        if ckpt_path.exists() and not bool(config.get("overwrite", False)):
            return ckpt_path

        W, bias = NeuronSampler(d=2, **fns_cfg.get("neuron_sampling", {})).sample(M_init)
        M_eff = int(W.shape[0])

        quad = _quad_cfg(fns_cfg, key="quadrature", fallback="gauss_quad")
        bquad = _quad_cfg(fns_cfg.get("boundary", {}), key="quadrature", fallback="boundary_quad")
        bcfg = BoundaryConfig(lambda_boundary=float(config.get("boundary_penalty_epsilon", 1.0)), quad=bquad)
        k = int(fns_cfg.get("reluk_k", 3))
        zero_grad = lambda pts: np.zeros((pts.shape[0], 2, 2), dtype=np.float64)

        with Timer() as t_asm:
            if solver_kind == "mass":
                res = assemble_stokes_2d(W, bias, zero_grad, k, quad, reg_lambda=0.0, boundary_cfg=bcfg, nu=float(config["nu"]))
                A = res.A
                rhs = res.b + self._boundary_rhs_mass(W, bias, k, bcfg, config.get("lid", {}))
            elif solver_kind == "direct_ls":
                res = assemble_stokes_2d_nomass(
                    W, bias, zero_grad, k, quad, boundary_cfg=BoundaryConfig(lambda_boundary=0.0, quad=bquad), nu=float(config["nu"])
                )
                A, rhs = self._append_boundary_rows_nomass(res.A, res.b, W, bias, k, bcfg, config.get("lid", {}))
            else:
                raise ValueError(f"Unknown solver: {solver_kind}")

        solver_cfg = fns_cfg.get(
            "linear_solver",
            {"name": "lstsq", "lstsq": {"rcond": None, "lapack_driver": "gelsd", "save_svd": True}},
        )
        solver = build_solver(solver_cfg)
        with Timer() as t_sol:
            a, sinfo = solver.solve(A, rhs)

        singular_values = None
        if isinstance(sinfo, dict):
            singular_values = sinfo.get("singular_values", None)

        meta = {
            "method": "fns",
            "solver": solver_kind,
            "linear_solver": solver_cfg,
            "M_init": M_init,
            "M": M_eff,
            "nu": float(config["nu"]),
            "boundary_penalty_epsilon": float(config.get("boundary_penalty_epsilon", 1.0)),
            "lid": config.get("lid", {}),
            "quadrature_train": fns_cfg.get("quadrature", fns_cfg.get("gauss_quad", {})),
            "boundary_quadrature_train": fns_cfg.get("boundary", {}).get("quadrature", fns_cfg.get("boundary_quad", {})),
            "timing_sec": {"assemble": float(t_asm.dt), "solve": float(t_sol.dt)},
            "residual_l2": float(np.linalg.norm(A @ a - rhs)),
            "svd_min": float(np.min(singular_values)) if singular_values is not None and len(singular_values) > 0 else None,
            "svd_max": float(np.max(singular_values)) if singular_values is not None and len(singular_values) > 0 else None,
        }

        arrays = {"W": W, "b": bias, "a": a}
        if singular_values is not None:
            arrays["singular_values"] = np.asarray(singular_values, dtype=np.float64)
        return self.ckpt.save_fns(ckpt_path, arrays=arrays, meta=meta)
