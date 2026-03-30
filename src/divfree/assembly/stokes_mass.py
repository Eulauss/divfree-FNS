from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from divfree.common.activation import relu, relu_k_derivative_scale
from divfree.common.quadrature import QuadratureConfig

from .boundary import BoundaryConfig, boundary_quadrature


@dataclass
class AssemblyResult:
    A: np.ndarray
    b: np.ndarray
    meta: Dict


def _linear_combo(pts: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    # pts: (n,d), W:(M,d), b:(M,)
    return pts @ W.T + b[None, :]


def _relu_k_second_derivative_scale(x: np.ndarray, k: int) -> np.ndarray:
    """Return k*(k-1)*ReLU(x)^(k-2) for k>=2."""
    k = int(k)
    if k < 2:
        raise ValueError("k must be >= 2 for H1 seminorm assembly")

    if k == 2:
        # Special case for k=2: the second derivative is the step function 2 * 1_{x>0}.
        # We explicitly use (x > 0) to avoid the 0^0=1 ambiguity in np.power.
        return 2.0 * (x > 0).astype(np.float64)
    
    # For k > 2, k-2 > 0, so np.power behaves correctly at 0.
    return float(k * (k - 1)) * np.power(relu(x), k - 2)


def _boundary_mass_2d(W: np.ndarray, b: np.ndarray, k: int, boundary_cfg: BoundaryConfig) -> np.ndarray:
    M = W.shape[0]
    A = np.zeros((M, M), dtype=np.float64)
    if boundary_cfg.lambda_boundary <= 0:
        return A
    w1 = W[:, 0]
    w2 = W[:, 1]
    for pts, wts in boundary_quadrature(d=2, quad=boundary_cfg.quad):
        z = _linear_combo(pts, W, b)
        S = relu_k_derivative_scale(z, k)
        sqrtw = np.sqrt(wts)[:, None]
        S_w = S * sqrtw
        Kb = S_w.T @ S_w
        A += Kb * (np.outer(w1, w1) + np.outer(w2, w2))
    return A


def _boundary_mass_3d(W: np.ndarray, b: np.ndarray, k: int, boundary_cfg: BoundaryConfig) -> np.ndarray:
    M = W.shape[0]
    dof = 3 * M
    A = np.zeros((dof, dof), dtype=np.float64)
    if boundary_cfg.lambda_boundary <= 0:
        return A
    w1 = W[:, 0]
    w2 = W[:, 1]
    w3 = W[:, 2]

    i12 = slice(0, M)
    i13 = slice(M, 2 * M)
    i23 = slice(2 * M, 3 * M)

    for pts, wts in boundary_quadrature(d=3, quad=boundary_cfg.quad):
        z = _linear_combo(pts, W, b)
        S = relu_k_derivative_scale(z, k)
        sqrtw = np.sqrt(wts)[:, None]
        S_w = S * sqrtw
        Kb = S_w.T @ S_w

        A[i12, i12] += Kb * (np.outer(w2, w2) + np.outer(w1, w1))
        A[i13, i13] += Kb * (np.outer(w3, w3) + np.outer(w1, w1))
        A[i23, i23] += Kb * (np.outer(w3, w3) + np.outer(w2, w2))

        B_12_13 = Kb * np.outer(w2, w3)
        A[i12, i13] += B_12_13
        A[i13, i12] += B_12_13.T

        B_12_23 = -Kb * np.outer(w1, w3)
        A[i12, i23] += B_12_23
        A[i23, i12] += B_12_23.T

        B_13_23 = Kb * np.outer(w1, w2)
        A[i13, i23] += B_13_23
        A[i23, i13] += B_13_23.T
    return A


def assemble_stokes_2d(
    W: np.ndarray,
    b: np.ndarray,
    target_grad_fn,
    k: int,
    quad: QuadratureConfig,
    reg_lambda: float,
    boundary_cfg: BoundaryConfig,
    nu: float,
) -> AssemblyResult:
    """
    Assemble the linear system for 2D Stokes variational fitting.

    We use the divergence-free basis
        Phi_i(x) = s_i(x) * (w2_i, -w1_i),   s_i = d/dt ReLU(t)^k, t=w_i·x+b_i,
    and minimize the energy functional
        J(u_M) = ∫_Ω (nu/2 * |∇u_M|^2 - f·u_M) dx + (λ/2) ∫_{∂Ω} |u_M|^2 ds,
    where f = -nu Δu^†, and we assemble b using the weak form
        b_i = nu * ∫_Ω ∇u^† : ∇Phi_i.

    The gradient term yields
        A_ij = nu * (w_i·w_j)^2 * ∫ s'_i s'_j,
        b_i  = nu * ∫ s'_i * (w2_i * (w_i·∇u1) - w1_i * (w_i·∇u2)),
    where s'_i is the derivative of s_i.
    The boundary penalty contributes the usual L2 boundary mass matrix.
    """
    M = W.shape[0]
    A = np.zeros((M, M), dtype=np.float64)
    rhs = np.zeros((M,), dtype=np.float64)

    w1 = W[:, 0].astype(np.float64)
    w2 = W[:, 1].astype(np.float64)
    wdot = np.outer(w1, w1) + np.outer(w2, w2)
    nu = float(nu)

    for pts, wts in quad.iter(d=2):
        z = _linear_combo(pts, W, b)
        S2 = _relu_k_second_derivative_scale(z, k)
        sqrtw = np.sqrt(wts)[:, None]
        S2_w = S2 * sqrtw
        K2 = S2_w.T @ S2_w

        grad_u = np.asarray(target_grad_fn(pts), dtype=np.float64)
        g1 = grad_u[:, 0, :] @ W.T
        g2 = grad_u[:, 1, :] @ W.T
        combo = w2[None, :] * g1 - w1[None, :] * g2
        rhs += nu * np.sum(wts[:, None] * S2 * combo, axis=0)

        A += nu * K2 * (wdot * wdot)

    if boundary_cfg.lambda_boundary > 0:
        A += float(boundary_cfg.lambda_boundary) * _boundary_mass_2d(W, b, k, boundary_cfg)

    if reg_lambda and reg_lambda > 0:
        A = A + float(reg_lambda) * np.eye(M)

    meta = {
        "dof": int(M),
        "M": int(M),
        "lambda_boundary": float(boundary_cfg.lambda_boundary),
        "boundary_quad": {"Nx": boundary_cfg.quad.Nx, "order": boundary_cfg.quad.order, "chunk_size": boundary_cfg.quad.chunk_size},
    }
    return AssemblyResult(A=A, b=rhs, meta=meta)


def assemble_stokes_3d(
    W: np.ndarray,
    b: np.ndarray,
    target_grad_fn,
    k: int,
    quad: QuadratureConfig,
    reg_lambda: float,
    boundary_cfg: BoundaryConfig,
    nu: float,
) -> AssemblyResult:
    """
    Assemble the linear system for 3D Stokes variational fitting.

    For each neuron i we use three divergence-free basis fields:
        Phi_{i,12} = s_i(x) * ( w2_i, -w1_i,  0 )
        Phi_{i,13} = s_i(x) * ( w3_i,  0 , -w1_i)
        Phi_{i,23} = s_i(x) * (  0 ,  w3_i, -w2_i)

    We minimize
        J(u_M) = ∫_Ω (nu/2 * |∇u_M|^2 - f·u_M) dx + (λ/2) ∫_{∂Ω} |u_M|^2 ds,
    with f = -nu Δu^†, and assemble b via the weak form using ∇u^†.

    The H1 seminorm gives block Gram entries
        A^{pq,rs}_{ij} = nu * (w_i·w_j) * (c_{i,pq}·c_{j,rs}) * ∫ s'_i s'_j,
    and the RHS entries
        b^{pq}_i = nu * ∫ s'_i * (c_{i,pq} · [w_i·∇u]) ,
    where c_{i,pq} is the vector in Phi_{i,pq} and [w_i·∇u] = (w_i·∇u1, w_i·∇u2, w_i·∇u3).
    """
    M = W.shape[0]
    dof = 3 * M
    A = np.zeros((dof, dof), dtype=np.float64)
    rhs = np.zeros((dof,), dtype=np.float64)

    w1 = W[:, 0].astype(np.float64)
    w2 = W[:, 1].astype(np.float64)
    w3 = W[:, 2].astype(np.float64)
    nu = float(nu)

    wdot = np.outer(w1, w1) + np.outer(w2, w2) + np.outer(w3, w3)

    i12 = slice(0, M)
    i13 = slice(M, 2 * M)
    i23 = slice(2 * M, 3 * M)

    for pts, wts in quad.iter(d=3):
        z = _linear_combo(pts, W, b)
        S2 = _relu_k_second_derivative_scale(z, k)
        sqrtw = np.sqrt(wts)[:, None]
        S2_w = S2 * sqrtw
        K2 = S2_w.T @ S2_w
        K2_wdot = K2 * wdot

        grad_u = np.asarray(target_grad_fn(pts), dtype=np.float64)
        g1 = grad_u[:, 0, :] @ W.T
        g2 = grad_u[:, 1, :] @ W.T
        g3 = grad_u[:, 2, :] @ W.T

        combo_12 = w2[None, :] * g1 - w1[None, :] * g2
        combo_13 = w3[None, :] * g1 - w1[None, :] * g3
        combo_23 = w3[None, :] * g2 - w2[None, :] * g3

        rhs[i12] += nu * np.sum(wts[:, None] * S2 * combo_12, axis=0)
        rhs[i13] += nu * np.sum(wts[:, None] * S2 * combo_13, axis=0)
        rhs[i23] += nu * np.sum(wts[:, None] * S2 * combo_23, axis=0)

        A[i12, i12] += nu * K2_wdot * (np.outer(w2, w2) + np.outer(w1, w1))
        A[i13, i13] += nu * K2_wdot * (np.outer(w3, w3) + np.outer(w1, w1))
        A[i23, i23] += nu * K2_wdot * (np.outer(w3, w3) + np.outer(w2, w2))

        B_12_13 = nu * K2_wdot * np.outer(w2, w3)
        A[i12, i13] += B_12_13
        A[i13, i12] += B_12_13.T

        B_12_23 = -nu * K2_wdot * np.outer(w1, w3)
        A[i12, i23] += B_12_23
        A[i23, i12] += B_12_23.T

        B_13_23 = nu * K2_wdot * np.outer(w1, w2)
        A[i13, i23] += B_13_23
        A[i23, i13] += B_13_23.T

    if boundary_cfg.lambda_boundary > 0:
        A += float(boundary_cfg.lambda_boundary) * _boundary_mass_3d(W, b, k, boundary_cfg)

    if reg_lambda and reg_lambda > 0:
        A = A + float(reg_lambda) * np.eye(dof)

    meta = {
        "dof": int(dof),
        "M": int(M),
        "lambda_boundary": float(boundary_cfg.lambda_boundary),
        "boundary_quad": {"Nx": boundary_cfg.quad.Nx, "order": boundary_cfg.quad.order, "chunk_size": boundary_cfg.quad.chunk_size},
    }
    return AssemblyResult(A=A, b=rhs, meta=meta)
