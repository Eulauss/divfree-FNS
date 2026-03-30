"""Metric evaluation for lid-driven cavity experiments using exact FEM polynomial evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from divfree.common.quadrature import QuadratureConfig
from divfree.assembly.boundary import boundary_quadrature
from divfree.tasks.stokes_homogeneous.runner import _predict_divfree_2d, _predict_divfree_2d_grad, _rate_table

from .checkpointing import CheckpointManager
from .fem_reference import ExactFEMEvaluator
from .lid_bc import build_lid_bc


def _quad_cfg(cfg: Dict[str, Any], key: str, fallback: str) -> QuadratureConfig:
    return QuadratureConfig(**cfg.get(key, cfg.get(fallback, {})))


@dataclass
class MetricsEvaluator2D:
    ckpt: CheckpointManager

    def evaluate(self, fns_ckpt: str, fem_ckpt: str, config: Dict[str, Any]) -> Dict[str, Any]:
        fns = self.ckpt.load(fns_ckpt)
        fem = self.ckpt.load(fem_ckpt)

        W, b, a = fns["arrays"]["W"], fns["arrays"]["b"], fns["arrays"]["a"]
        fem_eval = ExactFEMEvaluator(
            dofs_array=fem["arrays"]["dofs_array"],
            mesh_N=int(np.asarray(fem["arrays"]["mesh_N"]).ravel()[0]),
            fem_order=int(np.asarray(fem["arrays"]["fem_order"]).ravel()[0]),
        )

        fns_cfg = config["fns"]
        quad = _quad_cfg(fns_cfg, key="evaluation", fallback="gauss_quad")
        bquad = _quad_cfg(fns_cfg.get("boundary", {}), key="evaluation", fallback="boundary_quad")
        k = int(fns_cfg.get("reluk_k", 3))

        lid_type = str(config.get("lid", {}).get("type", "smooth")).lower()
        delta = float(fns_cfg.get("inner_evaluate", {}).get("delta", 0.1))

        err_l2 = err_h1 = div_fns2 = div_ref2 = 0.0
        err_l2_in = err_h1_in = 0.0

        for pts, wts in quad.iter(d=2):
            uf = _predict_divfree_2d(pts, W, b, a, k)
            gf = _predict_divfree_2d_grad(pts, W, b, a, k)
            ur, gr = fem_eval.evaluate_at_points(pts)

            du = uf - ur
            dg = gf - gr
            p_l2 = np.sum(du * du, axis=1)
            p_h1 = np.sum(dg * dg, axis=(1, 2))

            err_l2 += float(np.sum(wts * p_l2))
            err_h1 += float(np.sum(wts * p_h1))
            div_fns2 += float(np.sum(wts * (gf[:, 0, 0] + gf[:, 1, 1]) ** 2))
            div_ref2 += float(np.sum(wts * (gr[:, 0, 0] + gr[:, 1, 1]) ** 2))

            if lid_type == "classic":
                inner = (
                    (pts[:, 0] >= -1.0 + delta)
                    & (pts[:, 0] <= 1.0 - delta)
                    & (pts[:, 1] >= -1.0 + delta)
                    & (pts[:, 1] <= 1.0 - delta)
                )
                if np.any(inner):
                    err_l2_in += float(np.sum(wts[inner] * p_l2[inner]))
                    err_h1_in += float(np.sum(wts[inner] * p_h1[inner]))

        lid = build_lid_bc(config.get("lid", {}))
        err_b = 0.0
        for pts, wts in boundary_quadrature(d=2, quad=bquad):
            uf = _predict_divfree_2d(pts, W, b, a, k)
            gb = lid.eval_on_boundary(pts)
            err_b += float(np.sum(wts * np.sum((uf - gb) ** 2, axis=1)))

        out = {
            "Err_L2": float(np.sqrt(err_l2)),
            "Err_H1": float(np.sqrt(err_h1)),
            "Err_bdry": float(np.sqrt(err_b)),
            "Div_fns": float(np.sqrt(div_fns2)),
            "Div_ref": float(np.sqrt(div_ref2)),
        }
        if lid_type == "classic":
            out["Err_L2_inner"] = float(np.sqrt(err_l2_in))
            out["Err_H1_inner"] = float(np.sqrt(err_h1_in))
            out["inner_delta"] = delta
        return out

    def attach_rates(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_solver: Dict[str, List[Dict[str, Any]]] = {}
        out: List[Dict[str, Any]] = []
        for rec in records:
            by_solver.setdefault(rec["solver"], []).append(rec)

        for solver, recs in by_solver.items():
            base_df = _rate_table([{"M": r["M"], "h1_err_rel": r["Err_H1"], "l2_err_rel": r["Err_L2"]} for r in recs])
            base_df = base_df.rename(columns={"h1_rate": "rate_H1", "l2_rate": "rate_L2"})
            m2row = {int(row.M): row for _, row in base_df.iterrows()}

            has_inner = all("Err_L2_inner" in r and "Err_H1_inner" in r for r in recs)
            m2inner = {}
            if has_inner:
                inner_df = _rate_table([{"M": r["M"], "h1_err_rel": r["Err_H1_inner"], "l2_err_rel": r["Err_L2_inner"]} for r in recs])
                inner_df = inner_df.rename(columns={"h1_rate": "rate_H1_inner", "l2_rate": "rate_L2_inner"})
                m2inner = {int(row.M): row for _, row in inner_df.iterrows()}

            sorted_recs = sorted(recs, key=lambda x: x["M"])
            rate_bdry = [np.nan]
            for i in range(1, len(sorted_recs)):
                e0, e1 = sorted_recs[i - 1]["Err_bdry"], sorted_recs[i]["Err_bdry"]
                m0, m1 = sorted_recs[i - 1]["M"], sorted_recs[i]["M"]
                rate_bdry.append(float(-np.log(e1 / e0) / np.log(m1 / m0)) if e0 > 0 and e1 > 0 and m0 > 0 and m1 > 0 else np.nan)

            for rb, rec in zip(rate_bdry, sorted_recs):
                row = m2row[int(rec["M"])]
                rec["rate_L2"] = float(row.rate_L2) if np.isfinite(row.rate_L2) else np.nan
                rec["rate_H1"] = float(row.rate_H1) if np.isfinite(row.rate_H1) else np.nan
                rec["rate_bdry"] = rb
                if has_inner:
                    irow = m2inner[int(rec["M"])]
                    rec["rate_L2_inner"] = float(irow.rate_L2_inner) if np.isfinite(irow.rate_L2_inner) else np.nan
                    rec["rate_H1_inner"] = float(irow.rate_H1_inner) if np.isfinite(irow.rate_H1_inner) else np.nan
                out.append(rec)

        return sorted(out, key=lambda z: (z["solver"], z["M"]))
