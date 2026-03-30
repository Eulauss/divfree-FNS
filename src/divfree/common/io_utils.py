from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .utils import ensure_dir, write_json, append_jsonl


def save_config(run_dir: Path, cfg: Dict[str, Any]) -> None:
    write_json(run_dir / "config_used.json", cfg)


def save_record(run_dir: Path, record: Dict[str, Any]) -> None:
    append_jsonl(run_dir / "results.jsonl", record)


def save_matrix(run_dir: Path, name: str, A: np.ndarray, b: np.ndarray) -> Path:
    mats_dir = ensure_dir(run_dir / "matrices")
    out = Path(mats_dir) / f"{name}.npz"
    np.savez_compressed(out, A=A, b=b)
    return out


def save_parameters(
    run_dir: Path,
    problem_name: str,
    M: int,
    W: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    k: int,
    problem_type: str,
) -> Path:
    ckpt_dir = ensure_dir(run_dir / "check_points")
    out = Path(ckpt_dir) / f"{problem_name}_M={int(M)}.npz"
    np.savez_compressed(out, W=W, b=b, a=a, k=int(k), problem_type=str(problem_type))
    return out


def _predict_scalar(pts: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    z = pts @ W.T + b[None, :]
    phi = np.maximum(z, 0.0)
    if int(k) != 1:
        phi = np.power(phi, int(k))
    return phi @ a


def _predict_divfree_2d(pts: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    z = pts @ W.T + b[None, :]
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


def _predict_divfree_3d(pts: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    M = W.shape[0]
    a12 = a[:M]
    a13 = a[M:2 * M]
    a23 = a[2 * M:3 * M]
    z = pts @ W.T + b[None, :]
    if int(k) == 1:
        S = (z > 0).astype(np.float64)
    else:
        relu = np.maximum(z, 0.0)
        S = float(k) * np.power(relu, int(k) - 1)
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


def _predict_divfree_2d_grad(pts: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
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


def _predict_divfree_3d_grad(pts: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
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


def _decode_problem_type(raw: Any) -> str:
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


class ModelCheckpoint:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        data = np.load(path)
        self.W = np.asarray(data["W"], dtype=np.float64)
        self.b = np.asarray(data["b"], dtype=np.float64)
        self.a = np.asarray(data["a"], dtype=np.float64)
        self.k = int(np.asarray(data["k"]).item())
        self.problem_type = _decode_problem_type(data["problem_type"])

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        problem_type = str(self.problem_type).lower()
        if problem_type in {"scalar1d", "scalar"}:
            return _predict_scalar(x, self.W, self.b, self.a, self.k)
        if problem_type in {"divfree2d", "stokes2d"}:
            return _predict_divfree_2d(x, self.W, self.b, self.a, self.k)
        if problem_type in {"divfree3d", "stokes3d"}:
            return _predict_divfree_3d(x, self.W, self.b, self.a, self.k)
        raise ValueError(f"Unknown problem type: {self.problem_type}")

    def predict_grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        problem_type = str(self.problem_type).lower()
        if problem_type == "stokes2d":
            return _predict_divfree_2d_grad(x, self.W, self.b, self.a, self.k)
        if problem_type == "stokes3d":
            return _predict_divfree_3d_grad(x, self.W, self.b, self.a, self.k)
        raise ValueError(f"predict_grad is only available for stokes problems, got {self.problem_type}")
