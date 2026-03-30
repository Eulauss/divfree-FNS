from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_k(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return relu(x)
    return np.power(relu(x), k)


def relu_k_minus_1(x: np.ndarray, k: int) -> np.ndarray:
    """Returns ReLU(x)^(k-1) for k>=1."""
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        # convention for s(x) = 1_{x>0} when k=1; caller usually handles.
        return (x > 0).astype(np.float64)
    return np.power(relu(x), k - 1)


def relu_k_derivative_scale(x: np.ndarray, k: int) -> np.ndarray:
    """
    For phi(x) = ReLU(x)^k, the derivative wrt x is:
        phi'(x) = k * ReLU(x)^(k-1) * 1_{x>0}.
    This function returns s(x) = k * ReLU(x)^(k-1) (with correct k=1 handling).
    """
    k = int(k)
    if k == 1:
        return (x > 0).astype(np.float64)
    return float(k) * np.power(relu(x), k - 1)
