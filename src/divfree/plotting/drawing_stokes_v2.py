#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
drawing-l2.py

Edit `drawing_config` in this file, then run:
  python drawing-stokes.py

Outputs:
  1) loglog loss figure: x=M, y=h1_err_rel
  2) stacked FE-style rate tables into one txt file, separated by 3 blank lines
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# User config (edit here)
# =========================
drawing_config = {
    # Put your csv paths here (different k)
    "csv_paths": [
        "outputs/stokes/2d_stokes/stokes2d-nomass-k=2/rate_table.csv",
        "outputs/stokes/2d_stokes/stokes2d-nomass-k=3/rate_table.csv",
        "outputs/stokes/2d_stokes/stokes2d-nomass-k=4/rate_table.csv",
        "outputs/stokes/2d_stokes/stokes2d-nomass-k=5/rate_table.csv",
    ],

    # Output directory
    "out_dir": "outputs/figures_and_tables/stokes/2d_stokes",

    # Output filenames
    "fig_name": "stokes_d=2-nomass.png",
    "table_name": "stokes_rate_tables.txt",

    # Log base for tables: "log10" or "ln"
    "table_log": "log10",

    # Plot options
    "linewidth": 2.0,
    "markersize": 5,
    "dpi": 300,

    # Reference line
    "reference_line": True,
    # Group reference-line configs by error key; each error can have multiple lines.
    # "reference_line_config": {
    #     "2": {
    #         "slope": -0.75,
    #         "intercept": 22,
    #         "x_range": [400, 2800]
    #     },
    #     "3": {
    #         "slope": -1.25,
    #         "intercept": 100,
    #         "x_range": [400, 2800]
    #     },
    #     "4": {
    #         "slope": -1.75,
    #         "intercept": 580,
    #         "x_range": [400, 2800]
    #     },
    #     "5": {
    #         "slope": -2.25,
    #         "intercept": 3800,
    #         "x_range": [400, 2800]
    #     },
    # }
        "reference_line_config": {
        "2": {
            "slope": -2/3,
            "intercept": 37,
            "x_range": [1200, 3500]
        },
        "3": {
            "slope": -1.0,
            "intercept": 150,
            "x_range": [1200, 3500]
        },
        "4": {
            "slope": -4/3,
            "intercept": 850,
            "x_range": [1200, 3500]
        },
        "5": {
            "slope": -5/3,
            "intercept": 5300,
            "x_range": [1200, 3500]
        },
    }
}


def _safe_log(x: np.ndarray, base: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError(f"log requires positive entries, but got min={x.min()}")
    if base == "log10":
        return np.log10(x)
    if base == "ln":
        return np.log(x)
    raise ValueError(f"Unknown table_log={base}, must be 'log10' or 'ln'")


def read_csv_basic(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {"M", "h1_err_rel", "problem", "k"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[{path}] Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["M", "h1_err_rel"]).copy()
    df["M"] = pd.to_numeric(df["M"], errors="coerce")
    df["h1_err_rel"] = pd.to_numeric(df["h1_err_rel"], errors="coerce")
    df = df.dropna(subset=["M", "h1_err_rel"]).copy()

    df = df.sort_values("M").reset_index(drop=True)
    return df


def infer_problem_type(df: pd.DataFrame, path: str) -> str:
    vals = df["problem"].dropna().unique()
    if len(vals) == 0:
        raise ValueError(f"[{path}] problem_type exists but has no valid values.")
    if len(vals) > 1:
        print(f"[WARN] [{path}] Multiple problem_type values: {vals}. Using {vals[0]}.")
    return str(vals[0])


def infer_k(df: pd.DataFrame, path: str) -> int:
    vals = pd.to_numeric(df["k"], errors="coerce").dropna().unique()
    if len(vals) == 0:
        raise ValueError(f"[{path}] k exists but has no valid numeric values.")
    if len(vals) > 1:
        print(f"[WARN] [{path}] Multiple k values: {vals}. Using {vals[0]}.")
    return int(vals[0])

# ==================== 最小改动：字体设置 ====================
plt.rcParams.update({
    "font.family": "serif",              # 使用衬线字体 (Times New Roman 等)
    "mathtext.fontset": "stix",          # 数学公式使用 STIX 字体 (与 Times 风格匹配)
    "text.usetex": False,                # False 使用 matplotlib 内置渲染 (无需安装 LaTeX)
})
# =========================================================
def plot_loglog_loss(dfs: List[Tuple[str, pd.DataFrame]], out_path: str) -> None:
    problem_type = infer_problem_type(dfs[0][1], dfs[0][0])
    reference_line_config = drawing_config.get("reference_line_config")

    plt.figure(figsize=(7.5, 5.5))
    ax = plt.gca()

    ks = [infer_k(df, p) for p, df in dfs]
    unique_ks_sorted = sorted(set(ks))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
    if not color_cycle:
        color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    k2color = {k: color_cycle[i % len(color_cycle)] for i, k in enumerate(unique_ks_sorted)}

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    cidx = 0

    for (path, df) in dfs:
        k = infer_k(df, path)
        x = df["M"].to_numpy(dtype=float)
        y = df["h1_err_rel"].to_numpy(dtype=float)

        mask = (x > 0) & (y > 0)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            print(f"[WARN] [{path}] No positive data left; skip plotting.")
            continue

        marker = markers[cidx % len(markers)]
        ax.loglog(
            x, y,
            marker=marker,
            linewidth=drawing_config["linewidth"],
            markersize=drawing_config["markersize"],
            label=f"k={k}",
            color=k2color[k],
        )

        
        # Reference line
        if bool(drawing_config.get("reference_line", False)):
            refer = reference_line_config[str(k)]

            slope = float(refer["slope"])
            intercept = float(refer["intercept"])
            x0, x1 = refer.get("x_range", [float(np.min(x)), float(np.max(x))])
            xr = np.logspace(np.log10(float(x0)), np.log10(float(x1)), 60)
            yr = intercept * (xr ** slope)
            ref_color = k2color[k]
            ax.loglog(xr, yr, "--", color=ref_color, alpha=0.45, linewidth=1.3)

        cidx += 1

    # =========================================================
    # MODIFIED SECTION: Added fontsize to title
    # =========================================================
    ax.set_xlabel("Number of neurons $n$", fontsize=14)
    ax.set_ylabel("Relative $\dot{H}^1$ seminorm error", fontsize=14)
    # ax.set_title("div-free $L^2$ approximation, $d=3$", fontsize=13)
    ax.grid(False)
    ax.legend()
    # =========================================================

    plt.tight_layout()
    plt.savefig(out_path, dpi=drawing_config["dpi"])
    plt.close()
    print(f"[OK] Saved figure: {out_path}")


def make_rate_table_text(df: pd.DataFrame, csv_path: str) -> str:
    problem_type = infer_problem_type(df, csv_path)
    k = infer_k(df, csv_path)

    required_cols = {"M", "h1_err_rel", "rate"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[{csv_path}] Missing required columns for rate table: {sorted(missing)}")

    M = pd.to_numeric(df["M"], errors="coerce")
    err = pd.to_numeric(df["h1_err_rel"], errors="coerce")
    rate = pd.to_numeric(df["rate"], errors="coerce")

    mask = M.notna() & err.notna() & (M > 0) & (err > 0)
    M = M[mask].astype(float).to_numpy()
    err = err[mask].astype(float).to_numpy()
    rate = rate[mask].to_numpy(dtype=float)

    log_base = drawing_config["table_log"]
    logM = _safe_log(M, log_base)
    logerr = _safe_log(err, log_base)

    col_logM = f"{log_base}(M)"
    col_logerr = f"{log_base}(h1_err_rel)"

    table_df = pd.DataFrame({
        "M": M.astype(int) if np.all(np.isclose(M, np.round(M))) else M,
        col_logM: logM,
        col_logerr: logerr,
        "rate": rate,
    })

    title = f"{problem_type} k = {k} rate table"

    def fmt(x):
        if isinstance(x, float) and np.isnan(x):
            return "nan"
        if isinstance(x, (float, np.floating)):
            return f"{x:.6f}"
        return str(x)

    body = table_df.to_string(index=False, formatters={c: fmt for c in table_df.columns})
    return title + "\n" + body


def write_stacked_tables(dfs: List[Tuple[str, pd.DataFrame]], out_path: str) -> None:
    blocks = []
    for (path, df) in dfs:
        blocks.append(make_rate_table_text(df, path))
    content = ("\n\n\n").join(blocks) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Saved tables: {out_path}")


def main():
    csv_paths = drawing_config["csv_paths"]
    if not csv_paths:
        raise ValueError("drawing_config['csv_paths'] is empty. Please fill your csv paths.")

    out_dir = drawing_config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    dfs: List[Tuple[str, pd.DataFrame]] = []
    for p in csv_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"CSV not found: {p}")
        df = read_csv_basic(p)
        dfs.append((p, df))

    # 1) Figure
    fig_path = os.path.join(out_dir, drawing_config["fig_name"])
    plot_loglog_loss(dfs, fig_path)

    # 2) Tables
    table_path = os.path.join(out_dir, drawing_config["table_name"])
    write_stacked_tables(dfs, table_path)


if __name__ == "__main__":
    main()
