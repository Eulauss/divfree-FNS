#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Draw GT/prediction/error flow figures for three checkpoints.

Usage:
    1) Fill FLOWFIG_CONFIG['ckpt_paths'] with 3 checkpoint paths.
    2) Run: python drawing-flowfigure.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from divfree.common.io_utils import ModelCheckpoint
from divfree.tasks.l2_approx.targets import DivFreeTarget2D
from divfree.tasks.stokes_homogeneous.targets import StokesTarget2D


FLOWFIG_CONFIG = {
    "ckpt_paths": [
        "outputs/l2_approx/2d_L2/divfree2d-nomass-k=2/check_points/divfree2d_M=51.npz",
        "outputs/l2_approx/2d_L2/divfree2d-nomass-k=2/check_points/divfree2d_M=51.npz",
        "outputs/l2_approx/2d_L2/divfree2d-nomass-k=2/check_points/divfree2d_M=51.npz",
    ],
    "out_dir": "outputs/flowfigures",
    "out_basename": "flowfield_gt_pred_err-d2k2-nomass-nopositive.png",
    "freq": 3.141592653589793, # 3.141592653589793 or 9.42477796076938
    "target_family": "l2_divfree2d",  # "l2_divfree2d" | "stokes2d"
    "target_kind": "stream_sin",       # only used for l2_divfree2d
    "domain": (-1.0, 1.0, -1.0, 1.0),
    "grid_N": 181,
    "quiver_stride": 9,
    "heat_cmap": "viridis",
    "err_cmap": "magma",
    "heat_alpha": 0.95,
    "err_mode": "abs",  # "abs" or "rel"
    "rel_eps": 1e-8,
}


def _validate_config(cfg: dict) -> List[Path]:
    ckpt_paths = cfg.get("ckpt_paths", [])
    if not isinstance(ckpt_paths, list) or len(ckpt_paths) != 3:
        raise ValueError("FLOWFIG_CONFIG['ckpt_paths'] must be a list of exactly 3 checkpoint paths.")

    cleaned = [str(p).strip() for p in ckpt_paths]
    if any(not p for p in cleaned):
        raise ValueError("FLOWFIG_CONFIG['ckpt_paths'] must contain 3 non-empty paths (small/med/large).")

    ckpts = [Path(p) for p in cleaned]
    for p in ckpts:
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    err_mode = str(cfg.get("err_mode", "abs")).lower()
    if err_mode not in {"abs", "rel"}:
        raise ValueError("FLOWFIG_CONFIG['err_mode'] must be 'abs' or 'rel'.")

    return ckpts


def _make_grid(domain: Tuple[float, float, float, float], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = domain
    x = np.linspace(xmin, xmax, int(n), dtype=np.float64)
    y = np.linspace(ymin, ymax, int(n), dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, xy


def _extract_ckpt_label(path: Path) -> str:
    m = re.search(r"M\s*=\s*(\d+)", path.stem)
    if m:
        return rf"$n={int(m.group(1))}$"
    return path.stem


def _build_u_pred(ckpt_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    model = ModelCheckpoint(ckpt_path)

    def u_pred(xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Expected xy shape (N, 2), got {pts.shape}")

        if not hasattr(model, "predict"):
            raise TypeError("ModelCheckpoint must provide a numpy-based predict(xy) interface.")
        out = model.predict(pts)

        out = np.asarray(out, dtype=np.float64)
        if out.ndim != 2 or out.shape[1] != 2:
            raise ValueError(f"u_pred must return shape (N, 2), got {out.shape}")
        return out

    return u_pred


def _compute_error(u_pred: np.ndarray, u_gt: np.ndarray, mode: str, rel_eps: float) -> np.ndarray:
    diff = np.linalg.norm(u_pred - u_gt, axis=1)
    if mode == "abs":
        return diff
    denom = np.linalg.norm(u_gt, axis=1) + float(rel_eps)
    return diff / denom


def _set_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")


def _draw_heat_quiver(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    u: np.ndarray,
    speed: np.ndarray,
    domain: Tuple[float, float, float, float],
    stride: int,
    cmap: str,
    alpha: float,
    vmin: float,
    vmax: float,
):
    im = ax.imshow(
        speed,
        extent=domain,
        origin="lower",
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    eps = 1e-12
    uu = u[:, 0].reshape(xx.shape)
    vv = u[:, 1].reshape(yy.shape)
    mag = np.sqrt(uu**2 + vv**2)
    u_dir = uu / (mag + eps)
    v_dir = vv / (mag + eps)

    s = max(1, int(stride))
    ax.quiver(
        xx[::s, ::s],
        yy[::s, ::s],
        u_dir[::s, ::s],
        v_dir[::s, ::s],
        color="white",
        pivot="middle",
        scale=28,
        width=0.004,
        headwidth=3.0,
        headlength=4.2,
        headaxislength=3.6,
    )
    _style_axis(ax)
    return im


def main() -> None:
    cfg = FLOWFIG_CONFIG
    ckpt_paths = _validate_config(cfg)
    _set_academic_style()

    domain = tuple(cfg["domain"])
    grid_n = int(cfg["grid_N"])
    xx, yy, xy = _make_grid(domain, grid_n)

    target_family = str(cfg.get("target_family", "l2_divfree2d")).lower()
    if target_family == "l2_divfree2d":
        target = DivFreeTarget2D(kind=str(cfg.get("target_kind", "stream_sin")), freq=cfg["freq"])
    elif target_family == "stokes2d":
        target = StokesTarget2D(freq=cfg["freq"])
    else:
        raise ValueError(f"Unknown target_family: {target_family}")
    u_gt = target.eval(xy)

    pred_fns = [_build_u_pred(p) for p in ckpt_paths]
    u_preds = [fn(xy) for fn in pred_fns]

    gt_speed = np.linalg.norm(u_gt, axis=1).reshape(xx.shape)
    pred_speeds = [np.linalg.norm(u, axis=1).reshape(xx.shape) for u in u_preds]

    err_mode = str(cfg["err_mode"]).lower()
    errors = [
        _compute_error(u_pred=u, u_gt=u_gt, mode=err_mode, rel_eps=float(cfg["rel_eps"])) for u in u_preds
    ]
    err_maps = [e.reshape(xx.shape) for e in errors]

    vel_vmin, vel_vmax = 0.0, float(np.max(gt_speed))
    err_vmin, err_vmax = 0.0, float(np.max(np.stack(err_maps, axis=0)))

    fig = plt.figure(figsize=(15.5, 12.0))
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=4,
        width_ratios=[1.05, 1.05, 1.05, 0.05],
        height_ratios=[1.05, 1.05, 1.05],
        figure=fig,
    )

    ax_gt = fig.add_subplot(gs[0, 0:3])
    im_vel = _draw_heat_quiver(
        ax_gt,
        xx,
        yy,
        u_gt,
        gt_speed,
        domain,
        stride=int(cfg["quiver_stride"]),
        cmap=str(cfg["heat_cmap"]),
        alpha=float(cfg["heat_alpha"]),
        vmin=vel_vmin,
        vmax=vel_vmax,
    )
    ax_gt.set_title("Ground Truth", fontsize=14)
    # ax_gt.set_xlabel(r"$x$")
    # ax_gt.set_ylabel(r"$y$")
    tick_interval = 0.5 
    ax_gt.xaxis.set_major_locator(MultipleLocator(tick_interval))
    ax_gt.yaxis.set_major_locator(MultipleLocator(tick_interval))

    ax_legend = fig.add_subplot(gs[0, 2])
    ax_legend.axis("off")
    # legend_handles = [
    #     Line2D([0], [0], color="white", lw=2.5, label=r"Direction ($\mathbf{u}/|\mathbf{u}|$)"),
    #     Line2D([0], [0], color="none", label=r"Color: $|\mathbf{u}|$"),
    # ]
    # ax_legend.legend(
    #     handles=legend_handles, 
    #     loc="center left",
    #     bbox_to_anchor=(1.5, 0.5),  # 锚定到 ax_legend 右侧外部 (x>1 表示超出 axes 范围)
    #     frameon=False, 
    #     fontsize=11
    # )
    cbar_vel = fig.colorbar(im_vel, ax=ax_legend, fraction=0.9, pad=0.06)
    cbar_vel.set_label(r"$|\mathbf{u}|$", fontsize=14)

    pred_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for i, ax in enumerate(pred_axes):
        _draw_heat_quiver(
            ax,
            xx,
            yy,
            u_preds[i],
            pred_speeds[i],
            domain,
            stride=int(cfg["quiver_stride"]),
            cmap=str(cfg["heat_cmap"]),
            alpha=float(cfg["heat_alpha"]),
            vmin=vel_vmin,
            vmax=vel_vmax,
        )
        ax.set_title(_extract_ckpt_label(ckpt_paths[i]), fontsize=14)

        ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ax.set_xticks(ticks)    
        ax.set_yticks(ticks)    
        ax.set_aspect('equal')

    ax_row2_pad = fig.add_subplot(gs[1, 3])
    ax_row2_pad.axis("off")

    err_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    im_err = None
    for i, ax in enumerate(err_axes):
        im_err = ax.imshow(
            err_maps[i],
            extent=domain,
            origin="lower",
            cmap=str(cfg["err_cmap"]),
            vmin=err_vmin,
            vmax=err_vmax,
            interpolation="nearest",
            aspect="equal",
        )
        _style_axis(ax)
        ax.set_title(f"Error ({_extract_ckpt_label(ckpt_paths[i])})", fontsize=14)
        ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ax.set_xticks(ticks)    
        ax.set_yticks(ticks)    
        ax.set_aspect('equal')

    ax_err_cbar = fig.add_subplot(gs[2, 3])
    cbar_err = fig.colorbar(im_err, cax=ax_err_cbar)
    label = r"$\|\mathbf{u}_{n}-\mathbf{u}^\dagger\|_2$" if err_mode == "abs" else r"Relative error"
    cbar_err.set_label(label, fontsize=14)

    fig.tight_layout()

    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_basename = str(cfg["out_basename"])
    if out_basename.lower().endswith(".png"):
        out_name = out_basename
    else:
        out_name = f"{out_basename}.png"
    out_path = out_dir / out_name

    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(str(out_path))


if __name__ == "__main__":
    main()
