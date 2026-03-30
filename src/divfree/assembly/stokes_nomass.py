from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from divfree.common.activation import relu_k_derivative_scale
from divfree.common.quadrature import QuadratureConfig

from .stokes_mass import _relu_k_second_derivative_scale
from .boundary import BoundaryConfig, boundary_quadrature


@dataclass
class AssemblyResult:
    A: np.ndarray
    b: np.ndarray
    meta: Dict


def _linear_combo(pts: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return pts @ W.T + b[None, :]


def _num_quad_points(d: int, quad: QuadratureConfig) -> int:
    return int((quad.Nx ** d) * (quad.order ** d))


def _num_boundary_points(d: int, quad: QuadratureConfig) -> int:
    return int(2 * d * (quad.Nx ** (d - 1)) * (quad.order ** (d - 1)))


def assemble_stokes_2d_nomass(
    W: np.ndarray,
    b: np.ndarray,
    target_grad_fn,
    k: int,
    quad: QuadratureConfig,
    boundary_cfg: BoundaryConfig,
    nu: float,
) -> AssemblyResult:
    """
    Assemble a rectangular Gauss-point system for 2D Stokes H1 fitting.

    We solve:
        min_a || A a - b ||_2^2
    where rows correspond to weighted grad(u) components at Gauss points,
    with weights sqrt(nu * w_q) and basis contributions from s''(t).
    Boundary penalty rows (if lambda_boundary > 0) append weighted u values
    with factor sqrt(lambda_boundary * w_q).
    """
    M = W.shape[0]
    n_total = _num_quad_points(d=2, quad=quad)
    boundary_points = _num_boundary_points(d=2, quad=boundary_cfg.quad) if boundary_cfg.lambda_boundary > 0 else 0
    rows_interior = 4 * n_total
    rows_boundary = 2 * boundary_points
    total_rows = rows_interior + rows_boundary

    A = np.zeros((total_rows, M), dtype=np.float64, order='F')
    rhs = np.zeros((total_rows,), dtype=np.float64)

    w1 = W[:, 0].astype(np.float64)
    w2 = W[:, 1].astype(np.float64)
    nu = float(nu)

    row = 0
    for pts, wts in quad.iter(d=2):
        n = pts.shape[0]
        z = _linear_combo(pts, W, b)
        S2 = _relu_k_second_derivative_scale(z, k)
        sqrtw = np.sqrt(wts)[:, None]
        weight = sqrtw * np.sqrt(nu)
        base = S2 * weight

        rows00 = slice(row, row + n)
        rows01 = slice(row + n, row + 2 * n)
        rows10 = slice(row + 2 * n, row + 3 * n)
        rows11 = slice(row + 3 * n, row + 4 * n)

        A[rows00, :] = base * (w2 * w1)[None, :]
        A[rows01, :] = base * (w2 * w2)[None, :]
        A[rows10, :] = base * (-w1 * w1)[None, :]
        A[rows11, :] = base * (-w1 * w2)[None, :]

        grad_u = np.asarray(target_grad_fn(pts), dtype=np.float64)
        rhs[rows00] = weight[:, 0] * grad_u[:, 0, 0]
        rhs[rows01] = weight[:, 0] * grad_u[:, 0, 1]
        rhs[rows10] = weight[:, 0] * grad_u[:, 1, 0]
        rhs[rows11] = weight[:, 0] * grad_u[:, 1, 1]

        row += 4 * n

    if boundary_cfg.lambda_boundary > 0:
        lam_weight = np.sqrt(float(boundary_cfg.lambda_boundary))
        for pts, wts in boundary_quadrature(d=2, quad=boundary_cfg.quad):
            n = pts.shape[0]
            z = _linear_combo(pts, W, b)
            S = relu_k_derivative_scale(z, k)
            sqrtw = np.sqrt(wts)[:, None]
            weight = sqrtw * lam_weight
            base = S * weight

            rows1 = slice(row, row + n)
            rows2 = slice(row + n, row + 2 * n)
            A[rows1, :] = base * w2[None, :]
            A[rows2, :] = base * (-w1[None, :])
            row += 2 * n

    meta = {
        "dof": int(M),
        "M": int(M),
        "rows": int(total_rows),
        "lambda_boundary": float(boundary_cfg.lambda_boundary),
        "boundary_quad": {
            "Nx": boundary_cfg.quad.Nx,
            "order": boundary_cfg.quad.order,
            "chunk_size": boundary_cfg.quad.chunk_size,
        },
    }
    return AssemblyResult(A=A, b=rhs, meta=meta)


def assemble_stokes_3d_nomass(
    W: np.ndarray,
    b: np.ndarray,
    target_grad_fn,
    k: int,
    quad: QuadratureConfig,
    boundary_cfg: BoundaryConfig,
    nu: float,
) -> AssemblyResult:
    """
    Assemble a rectangular Gauss-point system for 3D Stokes H1 fitting.

    We solve:
        min_a || A a - b ||_2^2
    using 9 rows per interior quadrature point for grad(u) components,
    weighted by sqrt(nu * w_q). Boundary penalty rows (if enabled) append
    weighted u components with sqrt(lambda_boundary * w_q).
    """
    M = W.shape[0]
    dof = 3 * M
    n_total = _num_quad_points(d=3, quad=quad)
    boundary_points = _num_boundary_points(d=3, quad=boundary_cfg.quad) if boundary_cfg.lambda_boundary > 0 else 0
    rows_interior = 9 * n_total
    rows_boundary = 3 * boundary_points
    total_rows = rows_interior + rows_boundary

    A = np.zeros((total_rows, dof), dtype=np.float64, order='F')
    rhs = np.zeros((total_rows,), dtype=np.float64)

    w1 = W[:, 0].astype(np.float64)
    w2 = W[:, 1].astype(np.float64)
    w3 = W[:, 2].astype(np.float64)
    nu = float(nu)

    col12 = slice(0, M)
    col13 = slice(M, 2 * M)
    col23 = slice(2 * M, 3 * M)

    row = 0
    for pts, wts in quad.iter(d=3):
        n = pts.shape[0]
        z = _linear_combo(pts, W, b)
        S2 = _relu_k_second_derivative_scale(z, k)
        sqrtw = np.sqrt(wts)[:, None]
        weight = sqrtw * np.sqrt(nu)
        base = S2 * weight

        rows00 = slice(row, row + n)
        rows01 = slice(row + n, row + 2 * n)
        rows02 = slice(row + 2 * n, row + 3 * n)
        rows10 = slice(row + 3 * n, row + 4 * n)
        rows11 = slice(row + 4 * n, row + 5 * n)
        rows12 = slice(row + 5 * n, row + 6 * n)
        rows20 = slice(row + 6 * n, row + 7 * n)
        rows21 = slice(row + 7 * n, row + 8 * n)
        rows22 = slice(row + 8 * n, row + 9 * n)

        A[rows00, col12] = base * (w2 * w1)[None, :]
        A[rows01, col12] = base * (w2 * w2)[None, :]
        A[rows02, col12] = base * (w2 * w3)[None, :]
        A[rows10, col12] = base * (-w1 * w1)[None, :]
        A[rows11, col12] = base * (-w1 * w2)[None, :]
        A[rows12, col12] = base * (-w1 * w3)[None, :]

        A[rows00, col13] = base * (w3 * w1)[None, :]
        A[rows01, col13] = base * (w3 * w2)[None, :]
        A[rows02, col13] = base * (w3 * w3)[None, :]
        A[rows20, col13] = base * (-w1 * w1)[None, :]
        A[rows21, col13] = base * (-w1 * w2)[None, :]
        A[rows22, col13] = base * (-w1 * w3)[None, :]

        A[rows10, col23] = base * (w3 * w1)[None, :]
        A[rows11, col23] = base * (w3 * w2)[None, :]
        A[rows12, col23] = base * (w3 * w3)[None, :]
        A[rows20, col23] = base * (-w2 * w1)[None, :]
        A[rows21, col23] = base * (-w2 * w2)[None, :]
        A[rows22, col23] = base * (-w2 * w3)[None, :]

        grad_u = np.asarray(target_grad_fn(pts), dtype=np.float64)
        rhs[rows00] = weight[:, 0] * grad_u[:, 0, 0]
        rhs[rows01] = weight[:, 0] * grad_u[:, 0, 1]
        rhs[rows02] = weight[:, 0] * grad_u[:, 0, 2]
        rhs[rows10] = weight[:, 0] * grad_u[:, 1, 0]
        rhs[rows11] = weight[:, 0] * grad_u[:, 1, 1]
        rhs[rows12] = weight[:, 0] * grad_u[:, 1, 2]
        rhs[rows20] = weight[:, 0] * grad_u[:, 2, 0]
        rhs[rows21] = weight[:, 0] * grad_u[:, 2, 1]
        rhs[rows22] = weight[:, 0] * grad_u[:, 2, 2]

        row += 9 * n

    if boundary_cfg.lambda_boundary > 0:
        lam_weight = np.sqrt(float(boundary_cfg.lambda_boundary))
        for pts, wts in boundary_quadrature(d=3, quad=boundary_cfg.quad):
            n = pts.shape[0]
            z = _linear_combo(pts, W, b)
            S = relu_k_derivative_scale(z, k)
            sqrtw = np.sqrt(wts)[:, None]
            weight = sqrtw * lam_weight
            base = S * weight

            rows1 = slice(row, row + n)
            rows2 = slice(row + n, row + 2 * n)
            rows3 = slice(row + 2 * n, row + 3 * n)

            A[rows1, col12] = base * w2[None, :]
            A[rows1, col13] = base * w3[None, :]

            A[rows2, col12] = base * (-w1[None, :])
            A[rows2, col23] = base * w3[None, :]

            A[rows3, col13] = base * (-w1[None, :])
            A[rows3, col23] = base * (-w2[None, :])

            row += 3 * n

    meta = {
        "dof": int(dof),
        "M": int(M),
        "rows": int(total_rows),
        "lambda_boundary": float(boundary_cfg.lambda_boundary),
        "boundary_quad": {
            "Nx": boundary_cfg.quad.Nx,
            "order": boundary_cfg.quad.order,
            "chunk_size": boundary_cfg.quad.chunk_size,
        },
    }
    return AssemblyResult(A=A, b=rhs, meta=meta)
