from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np

from divfree.common.quadrature import QuadratureConfig


@dataclass(frozen=True)
class BoundaryConfig:
    lambda_boundary: float = 1.0
    quad: QuadratureConfig = QuadratureConfig()


def boundary_measure(d: int) -> float:
    """Surface measure of [-1,1]^d."""
    d = int(d)
    if d < 1:
        raise ValueError("d must be >= 1")
    return float(d * (2.0 ** d))


def boundary_piecewise_gq_generator(
    d: int, quad: QuadratureConfig
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Composite Gauss quadrature on the boundary of [-1,1]^d.

    Uses tensor-product Gauss on each (d-1)-dimensional face and yields chunks.
    """
    d = int(d)
    if d < 1:
        raise ValueError("d must be >= 1")
    if d == 1:
        pts = np.array([[-1.0], [1.0]], dtype=np.float64)
        wts = np.ones((2,), dtype=np.float64)
        yield pts, wts
        return

    for pts, wts in quad.iter(d=d - 1):
        n = pts.shape[0]
        for axis in range(d):
            for sign in (-1.0, 1.0):
                full = np.zeros((n, d), dtype=np.float64)
                before = pts[:, :axis]
                after = pts[:, axis:]
                if axis > 0:
                    full[:, :axis] = before
                if axis < d - 1:
                    full[:, axis + 1 :] = after
                full[:, axis] = sign
                yield full, wts


def boundary_quadrature(d: int, quad: QuadratureConfig) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Iterator for Gauss quadrature on boundary faces."""
    return boundary_piecewise_gq_generator(d=d, quad=quad)
