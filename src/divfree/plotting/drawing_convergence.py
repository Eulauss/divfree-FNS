#!/usr/bin/env python3
"""Draw publication-style convergence curves from stokes_lid state/result records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "text.usetex": False,
    }
)


drawing_config: Dict[str, Any] = {
    "input_path": "outputs/figures_and_tables/stokes_lid/result_fem&cos2-k=2,3.json",
    "output_dir": "outputs/figures_and_tables/fem&cos2-k=2,3",
    "error_keys": ["Err_L2", "Err_H1", "Err_bdry", "Err_L2_inner", "Err_H1_inner"],
    "x_key": "M",
    "x_label": r"degree of freedom(DOF) $n$",
    "dpi": 300,
    "reference_line": False,
    # Group reference-line configs by error key; each error can have multiple lines.
    "reference_line_config": {
        "Err_L2": [
            {
                "curve_id": "FNS_k2",
                "slope": -1.25,
                "intercept": 50,
                "x_range": [200, 1200],
                "color": "#8aa1c1",
            },
            {
                "curve_id": "FNS_k3",
                "slope": -1.75,
                "intercept": 100,
                "x_range": [200, 1200],
                "color": "#ADAA4CF1",
            },

        ],
    },
}



def _load_records(path: Path) -> List[Dict[str, Any]]:
    """Load records from *.json state files (supports legacy/result formats)."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "records" in data:
        return list(data["records"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unsupported record format in {path}")


def _solver_family(rec: Dict[str, Any]) -> str:
    """Normalize solver naming to FEM or FNS family labels."""
    s = str(rec.get("solver", "")).lower()
    if s == "fem_p2p1":
        return "FEM"
    if s in {"mass", "direct_ls", "fns_mass", "fns_direct_ls"}:
        return "FNS"
    return "OTHER"


def _get_x(rec: Dict[str, Any], x_key: str) -> float:
    """Read x-axis value with backward-compatible fallbacks."""
    if x_key in rec and rec[x_key] is not None:
        return float(rec[x_key])
    if "DOF" in rec and rec["DOF"] is not None:
        return float(rec["DOF"])
    return float(rec.get("M", np.nan))


def _collect_groups(records: List[Dict[str, Any]], error_key: str, x_key: str) -> Tuple[List[Tuple[str, str, np.ndarray, np.ndarray]], Dict[str, List[Dict[str, Any]]]]:
    """Group records for plotting and collect per-error reference-line configs by curve id."""
    groups: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for rec in records:
        if error_key not in rec:
            continue
        x = _get_x(rec, x_key)
        y = float(rec[error_key])
        if not np.isfinite(x) or not np.isfinite(y) or x <= 0 or y <= 0:
            continue

        fam = _solver_family(rec)
        if fam == "FEM":
            gid = ("FEM", "FEM")
        elif fam == "FNS":
            k = rec.get("k", None)
            k = int(k) if k is not None else -1
            gid = (f"FNS_k{k}", f"FNS, k={k}")
        else:
            continue

        groups.setdefault(gid, []).append((x, y))

    out: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    for (curve_id, label), xy in groups.items():
        xy = sorted(xy, key=lambda t: t[0])
        xs = np.array([t[0] for t in xy], dtype=float)
        ys = np.array([t[1] for t in xy], dtype=float)
        out.append((curve_id, label, xs, ys))

    raw_refs = drawing_config.get("reference_line_config", {}).get(error_key, [])
    refs_by_curve: Dict[str, List[Dict[str, Any]]] = {}
    for spec in raw_refs:
        cid = str(spec.get("curve_id", ""))
        if cid:
            refs_by_curve.setdefault(cid, []).append(spec)

    return out, refs_by_curve


def _draw_one_error(records: List[Dict[str, Any]], error_key: str, out_dir: Path) -> Path:
    """Draw one error-key curve figure in log-log scale."""
    groups, refs_by_curve = _collect_groups(records, error_key=error_key, x_key=str(drawing_config.get("x_key", "M")))
    if not groups:
        raise RuntimeError(f"No valid data for {error_key}")

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8), constrained_layout=True)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    cidx = 0

    y_label_mapping  = {
        "Err_L2" : "$L^2$ error",
        "Err_H1" : "$\dot{H}^1$ seminorm error",
        "Err_bdry" : "boundary $L^2$ error",
        "Err_L2_inner": "inner $L^2$ error",
        "Err_H1_inner": "inner $\dot{H}^1$ seminorm error"
    }

    for curve_id, label, x, y in groups:
        if curve_id == "FEM":
            line, = ax.loglog(x, y, "--", color="#555555", alpha=0.8, marker="*", linewidth=2.0, markersize=8, label="FEM reference")
        else:
            color = color_cycle[cidx % len(color_cycle)]
            marker = markers[cidx % len(markers)]
            line, = ax.loglog(x, y, "-", color=color, marker=marker, linewidth=2.0, markersize=5, label=label)
            cidx += 1

        if bool(drawing_config.get("reference_line", False)):
            for spec in refs_by_curve.get(curve_id, []):
                slope = float(spec["slope"])
                intercept = float(spec["intercept"])
                x0, x1 = spec.get("x_range", [float(np.min(x)), float(np.max(x))])
                xr = np.logspace(np.log10(float(x0)), np.log10(float(x1)), 60)
                yr = intercept * (xr ** slope)
                ref_color = str(spec.get("color", line.get_color()))
                ax.loglog(xr, yr, "--", color=ref_color, alpha=0.45, linewidth=1.3, label=f"slope={slope:g}")

    ax.set_xlabel(str(drawing_config.get("x_label", r"degree of freedom(DOF) $n$")), fontsize=14)
    ax.set_ylabel(y_label_mapping[error_key], fontsize=14)
    ax.set_facecolor("white")
    ax.grid(False)
    ax.legend(fontsize=9)

    out_path = out_dir / f"convergence_{error_key}.png"
    fig.savefig(out_path, dpi=int(drawing_config.get("dpi", 300)))
    plt.close(fig)
    return out_path


def main() -> None:
    """Load records and generate one figure per configured error key."""
    input_path = Path(str(drawing_config["input_path"]))
    out_dir = Path(str(drawing_config["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(input_path)
    for err in drawing_config.get("error_keys", []):
        try:
            out = _draw_one_error(records, err, out_dir)
            print(f"[OK] saved: {out}")
        except RuntimeError as exc:
            print(f"[WARN] skip {err}: {exc}")


if __name__ == "__main__":
    main()
