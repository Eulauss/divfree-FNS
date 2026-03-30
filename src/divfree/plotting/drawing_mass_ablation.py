#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
drawing_mass_ablation.py

Reads two CSV files (Direct LS and Normal Eq) and produces:
  1) Convergence rate log-log plot (l2_err_rel vs M)
  2) Condition number plot (kappa2 vs M) - with dashed lines for unstable regions

Edit the CSV paths below before running:
  python drawing_mass_ablation.py
"""

from __future__ import annotations

import ast
import json
import os
from typing import Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# =========================
# User config (edit here)
# =========================
DIRECT_LS_CSV = (
    "outputs/mass_ablation/divfree2d-nomass-k=4/rate_table.csv"
)
NORMAL_EQ_CSV = (
    "outputs/mass_ablation/divfree2d-mass-k=4/rate_table.csv"
)

OUT_DIR = "outputs/mass_ablation"
CONV_FIG_NAME = "mass_ablation_convergence.png"
COND_FIG_NAME = "mass_ablation_condition.png"

# Reference line: y = C * x^(-2.25)
REF_RATE = 2.25
REF_INTERCEPT_C = 100.0  # Adjust this to vertically align the reference line

# Machine Precision Threshold
MACHINE_EPSILON = 2.220446049250313e-16
COND_THRESHOLD = 1.0 / MACHINE_EPSILON

# Plot styling
FIGSIZE = (7.6, 5.6)
DPI = 300
MARKERSIZE = 8
LINEWIDTH = 2.5
FONT_SIZES = {
    "title": 13,
    "label": 14,
    "legend": 13,
    "ticks": 14,
}

STYLE_MAP: Dict[str, Dict[str, str]] = {
    "Direct LS": {"color": "#1f77b4", "marker": "o"},
    "Normal Eq": {"color": "#ff7f0e", "marker": "s"},
}


def _parse_solver_info(value: str) -> Dict[str, float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}

    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
    return {}


def _load_data(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        # Create dummy data for demonstration if file doesn't exist
        print(f"Warning: File {path} not found.")
        return pd.DataFrame() 

    df = pd.read_csv(path)

    required_cols = {"M", "l2_err_rel", "solver_info"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[{path}] Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["M", "l2_err_rel", "solver_info"]).copy()
    df["M"] = pd.to_numeric(df["M"], errors="coerce")
    df["l2_err_rel"] = pd.to_numeric(df["l2_err_rel"], errors="coerce")

    solver_info = df["solver_info"].apply(_parse_solver_info)
    df["singular_values_max"] = solver_info.apply(lambda x: x.get("singular_values_max"))
    df["singular_values_min"] = solver_info.apply(lambda x: x.get("singular_values_min"))

    df["singular_values_max"] = pd.to_numeric(df["singular_values_max"], errors="coerce")
    df["singular_values_min"] = pd.to_numeric(df["singular_values_min"], errors="coerce")

    df = df.dropna(subset=["M", "l2_err_rel", "singular_values_max", "singular_values_min"]).copy()
    df = df[(df["M"] > 0) & (df["l2_err_rel"] > 0) & (df["singular_values_min"] > 0)].copy()
    df["kappa2"] = df["singular_values_max"] / df["singular_values_min"]

    return df.sort_values("M").reset_index(drop=True)


def _set_axis_style(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=FONT_SIZES["label"])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
    ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["ticks"])
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

# ==================== 最小改动：字体设置 ====================
plt.rcParams.update({
    "font.family": "serif",              # 使用衬线字体 (Times New Roman 等)
    "mathtext.fontset": "stix",          # 数学公式使用 STIX 字体 (与 Times 风格匹配)
    "text.usetex": False,                # False 使用 matplotlib 内置渲染 (无需安装 LaTeX)
})
# =========================================================
def plot_convergence(data_map: Dict[str, pd.DataFrame], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    valid_data_exists = False
    for label, df in data_map.items():
        if df.empty: continue
        valid_data_exists = True
        style = STYLE_MAP.get(label, {"color": "black", "marker": "x"})
        ax.loglog(
            df["M"],
            df["l2_err_rel"],
            label=label,
            color=style["color"],
            marker=style["marker"],
            markersize=MARKERSIZE,
            linewidth=LINEWIDTH,
        )

    if valid_data_exists:
        # Calculate reference line based on the first dataset
        first_df = next(iter(data_map.values()))
        if not first_df.empty:
            ref_x = np.linspace(first_df["M"].min(), first_df["M"].max(), 100)
            ref_y = REF_INTERCEPT_C * (ref_x ** (-REF_RATE))
            ax.loglog(ref_x, ref_y, linestyle="--", color="0.6", linewidth=1.8, label=f"Ref: $O(n^{{-{REF_RATE}}})$")

    _set_axis_style(
        ax,
        xlabel="Number of neurons $n$",
        ylabel="Relative $L^2$ error",
        title="Convergence",
    )
    ax.legend(loc='best', fontsize=FONT_SIZES["legend"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"[OK] Saved convergence plot: {out_path}")


def plot_condition(data_map: Dict[str, pd.DataFrame], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for label, df in data_map.items():
        if df.empty: continue
        style = STYLE_MAP.get(label, {"color": "black", "marker": "x"})
        
        # Check for unstable points
        unstable_mask = df["kappa2"] > COND_THRESHOLD
        
        if not unstable_mask.any():
            # Case 1: All points are stable -> Simple plot
            ax.loglog(
                df["M"],
                df["kappa2"],
                label=label,
                color=style["color"],
                marker=style["marker"],
                markersize=MARKERSIZE,
                linewidth=LINEWIDTH,
            )
        else:
            # Case 2: Some points are unstable -> Split the line
            # Find the index of the first unstable point
            first_unsafe_idx = unstable_mask.idxmax()
            
            # The split point is one index BEFORE the first unsafe point
            # to make the segment connecting to the unsafe point dashed.
            split_idx = max(0, first_unsafe_idx - 1)
            
            # --- Reliable Part (Solid) ---
            # Plot from start up to split_idx (inclusive)
            if split_idx >= 0:
                ax.loglog(
                    df["M"].iloc[:split_idx+1],
                    df["kappa2"].iloc[:split_idx+1],
                    label=label, # Legend only on the solid part
                    color=style["color"],
                    marker=style["marker"],
                    markersize=MARKERSIZE,
                    linewidth=LINEWIDTH,
                    linestyle="-"
                )
            
            # --- Unreliable Part (Dashed & Lighter) ---
            # Plot from split_idx to end
            # This ensures the line is continuous but style changes at split_idx
            ax.loglog(
                df["M"].iloc[split_idx:],
                df["kappa2"].iloc[split_idx:],
                label=None, # No duplicate legend
                color=style["color"],
                marker=style["marker"],
                markersize=MARKERSIZE,
                linewidth=LINEWIDTH,
                linestyle="--",
                alpha=0.5 # Lighter color
            )

    # Add threshold line for reference (optional, helps visualization)
    ax.axhline(y=COND_THRESHOLD, color='r', linestyle=':', alpha=0.3, linewidth=1, label=r"$1/\epsilon_{mach}$")

    _set_axis_style(
        ax,
        xlabel="Number of neurons $n$",
        ylabel=r"Condition number $\kappa_2$",
        title="Condition Number",
    )
    ax.legend(loc='lower right', fontsize=FONT_SIZES["legend"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"[OK] Saved condition plot: {out_path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        direct_df = _load_data(DIRECT_LS_CSV)
        normal_df = _load_data(NORMAL_EQ_CSV)
    except FileNotFoundError as e:
        print(e)
        return

    data_map = {
        "Direct LS": direct_df,
        "Normal Eq": normal_df,
    }

    plot_convergence(data_map, os.path.join(OUT_DIR, CONV_FIG_NAME))
    plot_condition(data_map, os.path.join(OUT_DIR, COND_FIG_NAME))


if __name__ == "__main__":
    main()