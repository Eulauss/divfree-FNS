"""Matplotlib plotting utilities for lid-driven cavity outputs (exact FEM evaluator)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from divfree.tasks.stokes_homogeneous.runner import _predict_divfree_2d, _predict_divfree_2d_grad

from .checkpointing import CheckpointManager
from .fem_reference import ExactFEMEvaluator

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "text.usetex": False,
    }
)


def _draw_heat_quiver(
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    title: str,
    quiver_stride: int,
    cmap: str,
    vmin: float,
    vmax: float,
):
    """Draw speed heatmap + normalized quiver arrows for a vector field U:(Ny,Nx,2)."""
    speed = np.linalg.norm(U, axis=2)
    im = ax.pcolormesh(X, Y, speed, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ux, uy = U[:, :, 0], U[:, :, 1]
    mag = np.sqrt(ux * ux + uy * uy)
    eps = 1e-12
    s = max(1, int(quiver_stride))
    ax.quiver(
        X[::s, ::s],
        Y[::s, ::s],
        (ux / (mag + eps))[::s, ::s],
        (uy / (mag + eps))[::s, ::s],
        color="white",
        pivot="middle",
        scale=28,
        width=0.004,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    return im


def _draw_heat_only(ax: plt.Axes, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, title: str, cmap: str, vmin: float, vmax: float):
    """Draw scalar heatmap panel for Z:(Ny,Nx)."""
    im = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_aspect("equal")
    return im


@dataclass
class Visualizer2D:
    ckpt: CheckpointManager

    def plot_flow(self, ref_ckpt: str, fns_ckpt: str, config: Dict[str, Any]) -> List[Path]:
        """Render split figure files for flow comparison, L2 error, and H1-seminorm error."""
        out_dir = Path(config["output_dir"]) / "figs"
        out_dir.mkdir(parents=True, exist_ok=True)

        ref = self.ckpt.load(ref_ckpt)
        fns = self.ckpt.load(fns_ckpt)
        fem_eval = ExactFEMEvaluator(
            dofs_array=ref["arrays"]["dofs_array"],
            mesh_N=int(np.asarray(ref["arrays"]["mesh_N"]).ravel()[0]),
            fem_order=int(np.asarray(ref["arrays"]["fem_order"]).ravel()[0]),
        )

        N_plot = int(config.get("plotting", {}).get("grid_N", 201))
        x = np.linspace(-1.0, 1.0, N_plot, dtype=np.float64)
        y = np.linspace(-1.0, 1.0, N_plot, dtype=np.float64)
        X, Y = np.meshgrid(x, y, indexing="xy")
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)

        u_ref, g_ref = fem_eval.evaluate_at_points(pts)
        u_ref = u_ref.reshape(N_plot, N_plot, 2)
        g_ref = g_ref.reshape(N_plot, N_plot, 2, 2)

        k = int(config["fns"].get("reluk_k", 3))
        W, b, a = fns["arrays"]["W"], fns["arrays"]["b"], fns["arrays"]["a"]
        u_fns = _predict_divfree_2d(pts, W, b, a, k).reshape(u_ref.shape)
        g_fns = _predict_divfree_2d_grad(pts, W, b, a, k).reshape(g_ref.shape)

        err_l2 = np.linalg.norm(u_fns - u_ref, axis=2)
        err_h1 = np.linalg.norm(g_fns - g_ref, axis=(2, 3))

        vel_vmax = max(float(np.linalg.norm(u_ref, axis=2).max()), float(np.linalg.norm(u_fns, axis=2).max()), 1e-12)
        l2_vmax = max(float(err_l2.max()), 1e-12)
        h1_vmax = max(float(err_h1.max()), 1e-12)
        qstride = int(config.get("plotting", {}).get("quiver_stride", 16))

        M_eff = int(fns["meta"].get("M", W.shape[0]))
        out_paths: List[Path] = []

        # file 1: flow comparison (reference + FNS)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        im0 = _draw_heat_quiver(axes[0], X, Y, u_ref, "Reference", qstride, "viridis", 0.0, vel_vmax)
        im1 = _draw_heat_quiver(axes[1], X, Y, u_fns, f"FNS($n={M_eff}$)", qstride, "viridis", 0.0, vel_vmax)
        plt.colorbar(im0, ax=axes[0], fraction=0.05)
        plt.colorbar(im1, ax=axes[1], fraction=0.05)
        p_flow = out_dir / "flow_comparison_reference_vs_fns.png"
        fig.savefig(p_flow, dpi=220)
        plt.close(fig)
        out_paths.append(p_flow)

        # file 2: L2 error heatmap
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2), constrained_layout=True)
        im = _draw_heat_only(ax, X, Y, err_l2, "$L^2$ pointwise error", "magma", 0.0, l2_vmax)
        plt.colorbar(im, ax=ax, fraction=0.05)
        p_l2 = out_dir / "error_heatmap_l2_pointwise.png"
        fig.savefig(p_l2, dpi=220)
        plt.close(fig)
        out_paths.append(p_l2)

        # file 3: H1-seminorm error heatmap
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2), constrained_layout=True)
        im = _draw_heat_only(ax, X, Y, err_h1, "$\dot{H}^1$-seminorm pointwise error", "plasma", 0.0, h1_vmax)
        plt.colorbar(im, ax=ax, fraction=0.05)
        p_h1 = out_dir / "error_heatmap_h1_seminorm_pointwise.png"
        fig.savefig(p_h1, dpi=220)
        plt.close(fig)
        out_paths.append(p_h1)

        lid_type = str(config.get("lid", {}).get("type", "smooth")).lower()

        if lid_type == "classic":
            delta = float(config["fns"].get("inner_evaluate", {}).get("delta", 0.1))
            inner = (X >= -1.0 + delta) & (X <= 1.0 - delta) & (Y >= -1.0 + delta) & (Y <= 1.0 - delta)
            err_l2_inner = np.where(inner, err_l2, np.nan)
            err_h1_inner = np.where(inner, err_h1, np.nan)

            # file 3: inner L2 error heatmap
            fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2), constrained_layout=True)
            im = _draw_heat_only(ax, X, Y, err_l2_inner, "inner $L^2$ pointwise error", "magma", 0.0, max(float(np.nanmax(err_l2_inner)), 1e-12))
            plt.colorbar(im, ax=ax, fraction=0.05)
            p_l2_inner = out_dir / "inner_error_heatmap_l2_pointwise.png"
            fig.savefig(p_l2_inner, dpi=220)
            plt.close(fig)
            out_paths.append(p_l2_inner)

            # file 4: inner H1-seminorm error heatmap
            fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2), constrained_layout=True)
            im = _draw_heat_only(ax, X, Y, err_h1_inner, "inner $\dot{H}^1$-seminorm pointwise error", "plasma", 0.0, max(float(np.nanmax(err_h1_inner)), 1e-12))
            plt.colorbar(im, ax=ax, fraction=0.05)
            p_h1_inner = out_dir / "inner_error_heatmap_h1_seminorm_pointwise.png"
            fig.savefig(p_h1_inner, dpi=220)
            plt.close(fig)
            out_paths.append(p_h1_inner)

        return out_paths

    def _plot_single_error(self, records: List[Dict[str, Any]], x_label: str, y_key: str, out: Path) -> Path:
        """Plot one error family on a separate log-log figure."""
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
        for solver in sorted({r["solver"] for r in records}):
            rr = sorted([r for r in records if r["solver"] == solver], key=lambda t: t.get("DOF", t["M"]))
            x = np.array([r.get("DOF", r["M"]) for r in rr], dtype=float)
            y = np.array([r[y_key] for r in rr], dtype=float)
            mask = (x > 0) & (y > 0)
            if np.any(mask):
                ax.loglog(x[mask], y[mask], "-o", label=solver)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_key)
        ax.set_facecolor("white")
        ax.grid(False)
        ax.legend(fontsize=8)
        fig.savefig(out, dpi=220)
        plt.close(fig)
        return out

    def plot_convergence(self, records: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Path]:
        """Plot split convergence figures: L2, H1, and boundary errors."""
        out_dir = Path(config["output_dir"]) / "figs"
        out_dir.mkdir(parents=True, exist_ok=True)

        p1 = self._plot_single_error(records, "DOF", "Err_L2", out_dir / "convergence_l2.png")
        p2 = self._plot_single_error(records, "DOF", "Err_H1", out_dir / "convergence_h1.png")
        p3 = self._plot_single_error(records, "DOF", "Err_bdry", out_dir / "convergence_boundary.png")
        return [p1, p2, p3]
