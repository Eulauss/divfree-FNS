from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from divfree.common.activation import relu_k, relu_k_derivative_scale
from divfree.common.quadrature import QuadratureConfig


@dataclass
class AssemblyResult:
    A: np.ndarray
    b: np.ndarray
    meta: Dict


def _linear_combo(pts: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return pts @ W.T + b[None, :]


def _num_quad_points(d: int, quad: QuadratureConfig) -> int:
    return int((quad.Nx ** d) * (quad.order ** d))


def assemble_scalar_nomass(
    W: np.ndarray,
    b: np.ndarray,
    target_fn,
    k: int,
    quad: QuadratureConfig,
) -> AssemblyResult:
    """
    Assemble a rectangular Gauss-point design matrix for scalar L2 fitting.

    We form the weighted least-squares system:
        min_a || A a - b ||_2^2
    where rows correspond to Gauss quadrature points and
        A[q, i] = sqrt(w_q) * phi_i(x_q),
        b[q]    = sqrt(w_q) * f(x_q).
    """
    M = W.shape[0]
    d = int(W.shape[1])
    n_total = _num_quad_points(d=d, quad=quad)
    A = np.zeros((n_total, M), dtype=np.float64, order='F')
    rhs = np.zeros((n_total,), dtype=np.float64)

    row = 0
    for pts, wts in quad.iter(d=d):
        n = pts.shape[0]
        z = _linear_combo(pts, W, b)
        Phi = relu_k(z, k)
        sqrtw = np.sqrt(wts)[:, None]
        A[row : row + n, :] = Phi * sqrtw
        f = target_fn(pts)
        f = np.asarray(f, dtype=np.float64).reshape(-1)
        rhs[row : row + n] = sqrtw[:, 0] * f
        row += n

    meta = {"dof": int(M), "rows": int(n_total)}
    return AssemblyResult(A=A, b=rhs, meta=meta)


def assemble_divfree_2d_nomass(
    W: np.ndarray,
    b: np.ndarray,
    target_vec_fn,
    k: int,
    quad: QuadratureConfig,
) -> AssemblyResult:
    """
    Assemble a rectangular Gauss-point system for 2D div-free L2 fitting.

    We solve for a in:
        min_a || A a - b ||_2^2,
    where each quadrature point contributes two rows for (u1, u2) and
        A = [sqrt(w) * S * w2; sqrt(w) * S * (-w1)],
        b = [sqrt(w) * v1; sqrt(w) * v2],
    with S = d/dt ReLU(t)^k at t = w·x + b.
    """
    M = W.shape[0]
    n_total = _num_quad_points(d=2, quad=quad)
    A = np.zeros((2 * n_total, M), dtype=np.float64, order = 'F')
    rhs = np.zeros((2 * n_total,), dtype=np.float64)

    w1 = W[:, 0].astype(np.float64)
    w2 = W[:, 1].astype(np.float64)

    row = 0
    for pts, wts in quad.iter(d=2):
        n = pts.shape[0]
        z = _linear_combo(pts, W, b)
        S = relu_k_derivative_scale(z, k)
        sqrtw = np.sqrt(wts)[:, None]
        S_w = S * sqrtw

        rows1 = slice(row, row + n)
        rows2 = slice(row + n, row + 2 * n)

        A[rows1, :] = S_w * w2[None, :]
        A[rows2, :] = S_w * (-w1[None, :])

        v = np.asarray(target_vec_fn(pts), dtype=np.float64)
        rhs[rows1] = sqrtw[:, 0] * v[:, 0]
        rhs[rows2] = sqrtw[:, 0] * v[:, 1]

        row += 2 * n

    meta = {"dof": int(M), "rows": int(2 * n_total), "M": int(M)}
    return AssemblyResult(A=A, b=rhs, meta=meta)


def assemble_divfree_3d_nomass(
    W: np.ndarray,
    b: np.ndarray,
    target_vec_fn,
    k: int,
    quad: QuadratureConfig,
) -> AssemblyResult:
    """
    Assemble a rectangular Gauss-point system for 3D div-free L2 fitting.

    We solve:
        min_a || A a - b ||_2^2
    with 3 rows per quadrature point (u1, u2, u3). The column blocks correspond
    to the antisymmetric potential coefficients (a12, a13, a23) and use the
    same ReLU-derivative weights as in the mass-matrix formulation.
    """
    M = W.shape[0]
    dof = 3 * M
    n_total = _num_quad_points(d=3, quad=quad)
    A = np.zeros((3 * n_total, dof), dtype=np.float64, order='F')
    rhs = np.zeros((3 * n_total,), dtype=np.float64)

    w1 = W[:, 0].astype(np.float64)
    w2 = W[:, 1].astype(np.float64)
    w3 = W[:, 2].astype(np.float64)

    row = 0
    for pts, wts in quad.iter(d=3):
        n = pts.shape[0]
        z = _linear_combo(pts, W, b)
        S = relu_k_derivative_scale(z, k)
        sqrtw = np.sqrt(wts)[:, None]
        S_w = S * sqrtw

        rows1 = slice(row, row + n)
        rows2 = slice(row + n, row + 2 * n)
        rows3 = slice(row + 2 * n, row + 3 * n)

        A[rows1, :M] = S_w * w2[None, :]
        A[rows1, M : 2 * M] = S_w * w3[None, :]
        A[rows2, :M] = S_w * (-w1[None, :])
        A[rows2, 2 * M :] = S_w * w3[None, :]
        A[rows3, M : 2 * M] = S_w * (-w1[None, :])
        A[rows3, 2 * M :] = S_w * (-w2[None, :])

        v = np.asarray(target_vec_fn(pts), dtype=np.float64)
        rhs[rows1] = sqrtw[:, 0] * v[:, 0]
        rhs[rows2] = sqrtw[:, 0] * v[:, 1]
        rhs[rows3] = sqrtw[:, 0] * v[:, 2]

        row += 3 * n

    meta = {"dof": int(dof), "rows": int(3 * n_total), "M": int(M)}
    return AssemblyResult(A=A, b=rhs, meta=meta)
