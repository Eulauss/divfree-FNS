#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FFT error spectrum visualizer for divfree2d checkpoints.

Two plotting modes (select via config):
  1) "spectrum_vs_k":  plot radial-averaged power spectrum E(|k|) for each ckpt,
                      overlaid as multiple curves on ONE axis.
  2) "bands_vs_n":    bin the frequency axis into bands, then plot band-averaged
                      power versus neuron count n, where each curve is a band.

Usage:
  1) Edit FREQLOSS_CONFIG at the top.
  2) Run: python drawing-freqloss.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from divfree.common.io_utils import ModelCheckpoint
from divfree.tasks.l2_approx.targets import DivFreeTarget2D as Target


# =========================
# User config (edit here)
# =========================
FREQLOSS_CONFIG = {
    # Choose plot mode:
    #   - "spectrum_vs_k": old logic, curves (different ckpts) over |k|
    #   - "bands_vs_n":   new logic, curves (different frequency bands) over n
    "plot_mode": "bands_vs_n",

    # checkpoints (you may put 7~9 here)
    "ckpt_paths": [
        "runs_nopositive/freq-pi/iter300seed42-Sd/iter300seed42sample-l2-2d-nomass-Sd/divfree2d-nomass-k=1-lstsq-nopositive/check_points/divfree2d_M=26.npz",
        "runs_nopositive/freq-pi/iter300seed42-Sd/iter300seed42sample-l2-2d-nomass-Sd/divfree2d-nomass-k=1-lstsq-nopositive/check_points/divfree2d_M=51.npz",
        "runs_nopositive/freq-pi/iter300seed42-Sd/iter300seed42sample-l2-2d-nomass-Sd/divfree2d-nomass-k=1-lstsq-nopositive/check_points/divfree2d_M=100.npz",
        "runs_nopositive/freq-pi/iter300seed42-Sd/iter300seed42sample-l2-2d-nomass-Sd/divfree2d-nomass-k=1-lstsq-nopositive/check_points/divfree2d_M=202.npz",
        "runs_nopositive/freq-pi/iter300seed42-Sd/iter300seed42sample-l2-2d-nomass-Sd/divfree2d-nomass-k=1-lstsq-nopositive/check_points/divfree2d_M=400.npz",
        "runs_nopositive/freq-pi/iter300seed42-Sd/iter300seed42sample-l2-2d-nomass-Sd/divfree2d-nomass-k=1-lstsq-nopositive/check_points/divfree2d_M=801.npz",
        "runs_nopositive/freq-pi/iter300seed42-Sd/iter300seed42sample-l2-2d-nomass-Sd/divfree2d-nomass-k=1-lstsq-nopositive/check_points/divfree2d_M=1604.npz",
        "runs_nopositive/freq-pi/iter300seed42-Sd/iter300seed42sample-l2-2d-nomass-Sd/divfree2d-nomass-k=1-lstsq-nopositive/check_points/divfree2d_M=3202.npz"
    ],

    "out_dir": "runs_figures/freqloss/divfree2d-freq1pi",
    "out_basename": "freqloss_radial_spectrum-d2k1-nomass-nopositive-bands_vs_n",

    # Target
    "freq": np.pi,  # e.g., np.pi or 3*np.pi
    "domain": (-1.0, 1.0, -1.0, 1.0),

    # Grid for FFT (must be uniform)
    "grid_N": 256,  # recommend 256/512

    # Error definition
    "err_mode": "vec",   # "vec" => FFT on components then sum powers; "mag" => FFT on |e|
    "rel_mode": False,   # if True: normalize pointwise by (|u_gt|+eps) BEFORE FFT
    "rel_eps": 1e-8,

    # Radial spectrum sampling (k -> radial)
    "n_radial_bins": 40,
    "use_angular_freq": False,  # False => cycles/unit; True => radians/unit (2*pi factor)
    "drop_dc": False,           # remove k=0 term (often dominates)
    "normalize_fft": True,      # divide FFT by (Nx*Ny) for comparable scale

    # Banding (only used when plot_mode="bands_vs_n")
    "n_freq_bands": 3,          # number of frequency bands = number of curves
    "band_stat": "mean",        # "mean" or "sum" within a band
    "band_edges": "log",     # "linear" or "log" spacing over |k| in [k_min,k_max]
    "band_kmin": None,          # None => auto (use >0 min if log, else 0)
    "band_kmax": None,          # None => auto from max(|k|)
    "xscale": "log",            # x-axis scaling for n: "log" or "linear"
    "yscale": "log",            # y-axis scaling for power: "log" or "linear"

    # Plot settings
    "loglog": True,             # used by plot_mode="spectrum_vs_k" only
    "dpi": 300,
    "figsize": (7.0, 4.8),
    "legend_ncol": 2,
}


# -------------------------
# helpers
# -------------------------
def _validate_config(cfg: dict) -> List[Path]:
    ckpt_paths = cfg.get("ckpt_paths", [])
    if not isinstance(ckpt_paths, list) or len(ckpt_paths) < 1:
        raise ValueError("FREQLOSS_CONFIG['ckpt_paths'] must be a non-empty list of checkpoint paths.")
    if len(ckpt_paths) > 20:
        raise ValueError("Too many checkpoints; please keep <= 20 to avoid clutter.")

    ckpts = [Path(str(p).strip()) for p in ckpt_paths]
    for p in ckpts:
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    err_mode = str(cfg.get("err_mode", "vec")).lower()
    if err_mode not in {"vec", "mag"}:
        raise ValueError("FREQLOSS_CONFIG['err_mode'] must be 'vec' or 'mag'.")

    plot_mode = str(cfg.get("plot_mode", "")).lower()
    if plot_mode not in {"spectrum_vs_k", "bands_vs_n"}:
        raise ValueError("FREQLOSS_CONFIG['plot_mode'] must be 'spectrum_vs_k' or 'bands_vs_n'.")

    band_edges = str(cfg.get("band_edges", "linear")).lower()
    if band_edges not in {"linear", "log"}:
        raise ValueError("FREQLOSS_CONFIG['band_edges'] must be 'linear' or 'log'.")

    band_stat = str(cfg.get("band_stat", "mean")).lower()
    if band_stat not in {"mean", "sum"}:
        raise ValueError("FREQLOSS_CONFIG['band_stat'] must be 'mean' or 'sum'.")

    return ckpts


def _set_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def _extract_n_value(path: Path) -> int:
    m = re.search(r"M\s*=\s*(\d+)", path.stem)
    if m:
        return int(m.group(1))
    # fallback: any digits
    m2 = re.search(r"(\d+)", path.stem)
    if m2:
        return int(m2.group(1))
    raise ValueError(f"Cannot parse n from checkpoint name: {path.name}")


def _extract_ckpt_label(path: Path) -> str:
    m = re.search(r"M\s*=\s*(\d+)", path.stem)
    if m:
        return rf"$n={int(m.group(1))}$"
    return path.stem


def _make_grid(domain: Tuple[float, float, float, float], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    xmin, xmax, ymin, ymax = domain
    # endpoint=False to better match periodic FFT assumption on [-1,1)^2
    x = np.linspace(xmin, xmax, int(n), endpoint=False, dtype=np.float64)
    y = np.linspace(ymin, ymax, int(n), endpoint=False, dtype=np.float64)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, xy, dx, dy


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


def _compute_error_field(
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    xx_shape: Tuple[int, int],
    err_mode: str,
    rel_mode: bool,
    rel_eps: float,
) -> np.ndarray:
    e = (u_pred - u_gt).reshape((*xx_shape, 2))
    if rel_mode:
        gt = u_gt.reshape((*xx_shape, 2))
        denom = np.linalg.norm(gt, axis=-1, keepdims=True) + float(rel_eps)
        e = e / denom

    if err_mode == "vec":
        return e
    return np.linalg.norm(e, axis=-1)


def _radial_average_power(
    power2d: np.ndarray,
    dx: float,
    dy: float,
    n_bins: int,
    use_angular: bool,
    drop_dc: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    power2d: (Ny, Nx) nonnegative
    returns (k_centers, radial_mean_power)
    """
    ny, nx = power2d.shape
    kx = np.fft.fftfreq(nx, d=dx)  # cycles / unit
    ky = np.fft.fftfreq(ny, d=dy)
    kxx, kyy = np.meshgrid(kx, ky, indexing="xy")
    kr = np.sqrt(kxx**2 + kyy**2)
    if use_angular:
        kr = 2.0 * np.pi * kr

    p = power2d.copy()
    if drop_dc:
        p[0, 0] = 0.0

    kr_flat = kr.ravel()
    p_flat = p.ravel()

    r_max = float(np.max(kr_flat))
    if r_max <= 0:
        raise ValueError("Max frequency radius is non-positive; check grid_N and spacing.")

    edges = np.linspace(0.0, r_max, int(n_bins) + 1)
    ids = np.digitize(kr_flat, edges) - 1
    ids = np.clip(ids, 0, int(n_bins) - 1)

    sums = np.bincount(ids, weights=p_flat, minlength=int(n_bins)).astype(np.float64)
    cnts = np.bincount(ids, minlength=int(n_bins)).astype(np.float64)
    means = sums / np.maximum(cnts, 1.0)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, means


def _compute_radial_spectrum_for_ckpt(
    ckpt: Path,
    xy: np.ndarray,
    xx_shape: Tuple[int, int],
    u_gt: np.ndarray,
    grid_n: int,
    dx: float,
    dy: float,
    cfg: dict,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Return (n, k_centers, radial_power) for one checkpoint."""
    n_val = _extract_n_value(ckpt)
    u_fn = _build_u_pred(ckpt)
    u_pred = u_fn(xy)

    err_field = _compute_error_field(
        u_pred=u_pred,
        u_gt=u_gt,
        xx_shape=xx_shape,
        err_mode=str(cfg["err_mode"]).lower(),
        rel_mode=bool(cfg["rel_mode"]),
        rel_eps=float(cfg["rel_eps"]),
    )

    # FFT power
    if err_field.ndim == 3:
        e1 = err_field[..., 0]
        e2 = err_field[..., 1]
        F1 = np.fft.fft2(e1)
        F2 = np.fft.fft2(e2)
        if cfg["normalize_fft"]:
            F1 = F1 / (grid_n * grid_n)
            F2 = F2 / (grid_n * grid_n)
        power = (np.abs(F1) ** 2) + (np.abs(F2) ** 2)
    else:
        Fs = np.fft.fft2(err_field)
        if cfg["normalize_fft"]:
            Fs = Fs / (grid_n * grid_n)
        power = (np.abs(Fs) ** 2)

    k, radial = _radial_average_power(
        power2d=power,
        dx=dx,
        dy=dy,
        n_bins=int(cfg["n_radial_bins"]),
        use_angular=bool(cfg["use_angular_freq"]),
        drop_dc=bool(cfg["drop_dc"]),
    )
    return n_val, k, radial


def _make_band_edges(k_min: float, k_max: float, n_bands: int, mode: str) -> np.ndarray:
    if n_bands < 1:
        raise ValueError("n_freq_bands must be >= 1.")
    if mode == "linear":
        return np.linspace(k_min, k_max, n_bands + 1)
    # log
    if k_min <= 0:
        raise ValueError("For log band edges, k_min must be > 0.")
    return np.geomspace(k_min, k_max, n_bands + 1)


# -------------------------
# plot modes
# -------------------------
def plot_spectrum_vs_k(cfg: dict, ckpt_paths: List[Path]) -> Path:
    domain = tuple(cfg["domain"])
    grid_n = int(cfg["grid_N"])
    xx, yy, xy, dx, dy = _make_grid(domain, grid_n)

    target = Target(freq=float(cfg["freq"]))
    u_gt = target.eval(xy)

    fig, ax = plt.subplots(1, 1, figsize=tuple(cfg["figsize"]))

    # sort by n for nicer legend ordering
    ckpt_paths = sorted(ckpt_paths, key=_extract_n_value)

    for ckpt in ckpt_paths:
        n_val, k, radial = _compute_radial_spectrum_for_ckpt(
            ckpt=ckpt,
            xy=xy,
            xx_shape=xx.shape,
            u_gt=u_gt,
            grid_n=grid_n,
            dx=dx,
            dy=dy,
            cfg=cfg,
        )

        label = _extract_ckpt_label(ckpt)

        if cfg.get("loglog", True):
            mask = (k > 0) & (radial > 0)
            ax.loglog(k[mask], radial[mask], linewidth=2.0, label=label)
        else:
            ax.plot(k, radial, linewidth=2.0, label=label)

    ax.set_xlabel(r"$|k|$" + (r" (rad/unit)" if cfg["use_angular_freq"] else r" (cycles/unit)"))
    ylab = "Radially averaged power"
    if cfg["rel_mode"]:
        ylab += " (rel error)"
    ax.set_ylabel(ylab)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False, fontsize=10, ncol=int(cfg.get("legend_ncol", 2)))
    fig.tight_layout()

    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg['out_basename']}-spectrum_vs_k.png"
    fig.savefig(out_path, dpi=int(cfg["dpi"]), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_bands_vs_n(cfg: dict, ckpt_paths: List[Path]) -> Path:
    domain = tuple(cfg["domain"])
    grid_n = int(cfg["grid_N"])
    xx, yy, xy, dx, dy = _make_grid(domain, grid_n)

    target = Target(freq=float(cfg["freq"]))
    u_gt = target.eval(xy)

    # compute spectra for all ckpts
    records = []  # list of (n, k, radial)
    for ckpt in ckpt_paths:
        records.append(
            _compute_radial_spectrum_for_ckpt(
                ckpt=ckpt,
                xy=xy,
                xx_shape=xx.shape,
                u_gt=u_gt,
                grid_n=grid_n,
                dx=dx,
                dy=dy,
                cfg=cfg,
            )
        )

    # sort by n
    records.sort(key=lambda t: t[0])
    n_list = np.array([t[0] for t in records], dtype=float)

    # determine global k-range
    all_k = np.concatenate([t[1] for t in records])
    k_max_auto = float(np.max(all_k))

    n_bands = int(cfg.get("n_freq_bands", 6))
    mode = str(cfg.get("band_edges", "linear")).lower()

    kmax = float(cfg["band_kmax"]) if cfg.get("band_kmax") is not None else k_max_auto

    if cfg.get("band_kmin") is not None:
        kmin = float(cfg["band_kmin"])
    else:
        if mode == "log":
            # choose smallest positive k across bins to avoid 0 in log space
            kpos = all_k[all_k > 0]
            if kpos.size == 0:
                raise ValueError("No positive frequencies found; cannot use log band edges.")
            kmin = float(np.min(kpos))
        else:
            kmin = 0.0

    if not (kmax > kmin):
        raise ValueError(f"Invalid band range: kmin={kmin}, kmax={kmax}")

    edges = _make_band_edges(kmin, kmax, n_bands, mode)

    band_stat = str(cfg.get("band_stat", "mean")).lower()
    if band_stat not in {"mean", "sum"}:
        raise ValueError("band_stat must be 'mean' or 'sum'.")

    # compute band-averaged power Y[i, b]
    Y = np.full((len(records), n_bands), np.nan, dtype=np.float64)
    for i, (_, k, radial) in enumerate(records):
        for b in range(n_bands):
            lo, hi = edges[b], edges[b + 1]
            if b == n_bands - 1:
                mask = (k >= lo) & (k <= hi)
            else:
                mask = (k >= lo) & (k < hi)
            vals = radial[mask]
            if vals.size == 0:
                continue
            Y[i, b] = float(np.mean(vals)) if band_stat == "mean" else float(np.sum(vals))

    fig, ax = plt.subplots(1, 1, figsize=tuple(cfg["figsize"]))

    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        label = rf"$|k|\in[{lo:.3g},{hi:.3g}]$"
        ax.plot(n_list, Y[:, b], marker="o", linewidth=2.0, markersize=4.5, label=label)

    # axis scaling
    xscale = str(cfg.get("xscale", "log")).lower()
    yscale = str(cfg.get("yscale", "log")).lower()
    ax.set_xscale("log" if xscale == "log" else "linear")
    ax.set_yscale("log" if yscale == "log" else "linear")

    ax.set_xlabel(r"neuron count $n$")
    ylab = "band-averaged radial power"
    if cfg["rel_mode"]:
        ylab += " (rel error)"
    ax.set_ylabel(ylab)

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False, fontsize=9, ncol=int(cfg.get("legend_ncol", 2)))
    fig.tight_layout()

    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg['out_basename']}-bands_vs_n.png"
    fig.savefig(out_path, dpi=int(cfg["dpi"]), bbox_inches="tight")
    plt.close(fig)
    return out_path


# -------------------------
# entry
# -------------------------
def main() -> None:
    cfg = FREQLOSS_CONFIG
    ckpt_paths = _validate_config(cfg)
    _set_academic_style()

    plot_mode = str(cfg.get("plot_mode", "spectrum_vs_k")).lower()
    if plot_mode == "spectrum_vs_k":
        out = plot_spectrum_vs_k(cfg, ckpt_paths)
    else:
        out = plot_bands_vs_n(cfg, ckpt_paths)

    print(str(out))


if __name__ == "__main__":
    main()
