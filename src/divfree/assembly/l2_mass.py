from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import numpy as np

from divfree.common.activation import relu_k, relu_k_derivative_scale
from divfree.common.quadrature import QuadratureConfig


@dataclass
class AssemblyResult:
    A: np.ndarray
    b: np.ndarray
    meta: Dict


def _linear_combo(pts: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    # pts: (n,d), W:(M,d), b:(M,)
    return pts @ W.T + b[None, :]


def assemble_scalar(
    W: np.ndarray,
    b: np.ndarray,
    target_fn,
    k: int,
    quad: QuadratureConfig,
    reg_lambda: float,
) -> AssemblyResult:
    """A_{ij} = ∫ phi_i phi_j, b_i = ∫ f phi_i over [-1,1]^d."""
    M = W.shape[0]
    A = np.zeros((M, M), dtype=np.float64)
    rhs = np.zeros((M,), dtype=np.float64)
    d = int(W.shape[1])

    for pts, wts in quad.iter(d=d):
        z = _linear_combo(pts, W, b)              # (n,M)
        Phi = relu_k(z, k)                        # (n,M)
        # weight
        sqrtw = np.sqrt(wts)[:, None]
        Phi_w = Phi * sqrtw
        A += Phi_w.T @ Phi_w
        f = target_fn(pts)                        # (n,) or (n,1)
        f = np.asarray(f, dtype=np.float64).reshape(-1)
        rhs += Phi.T @ (wts * f)

    if reg_lambda and reg_lambda > 0:
        A = A + float(reg_lambda) * np.eye(M)

    meta = {"dof": int(M)}
    return AssemblyResult(A=A, b=rhs, meta=meta)


def assemble_divfree_2d(
    W: np.ndarray,
    b: np.ndarray,
    target_vec_fn,
    k: int,
    quad: QuadratureConfig,
    reg_lambda: float,
) -> AssemblyResult:
    """
    Uses basis Phi_i(x) = s_i(x) * (w2_i, -w1_i), where s_i = d/dt ReLU(t)^k at t=w·x+b.
    A_{ij} = ∫ Phi_i·Phi_j = K_{ij} * (w_i · w_j)  (in R^2),
    b_i = ∫ Phi_i·v = w2_i u1_i - w1_i u2_i, where u_r_i = ∫ s_i v_r.
    """
    M = W.shape[0]
    K = np.zeros((M, M), dtype=np.float64)
    u1 = np.zeros((M,), dtype=np.float64)
    u2 = np.zeros((M,), dtype=np.float64)

    w1 = W[:, 0].astype(np.float64)
    w2 = W[:, 1].astype(np.float64)

    for pts, wts in quad.iter(d=2):
        z = _linear_combo(pts, W, b)                      # (n,M)
        S = relu_k_derivative_scale(z, k)                 # (n,M)
        sqrtw = np.sqrt(wts)[:, None]
        S_w = S * sqrtw
        K += S_w.T @ S_w

        v = np.asarray(target_vec_fn(pts), dtype=np.float64)  # (n,2)
        u1 += S.T @ (wts * v[:, 0])
        u2 += S.T @ (wts * v[:, 1])

    # A = K * (outer(w1,w1)+outer(w2,w2))
    A = K * (np.outer(w1, w1) + np.outer(w2, w2))
    rhs = w2 * u1 - w1 * u2

    if reg_lambda and reg_lambda > 0:
        A = A + float(reg_lambda) * np.eye(M)

    meta = {"dof": int(M), "M": int(M)}
    return AssemblyResult(A=A, b=rhs, meta=meta)


def assemble_divfree_3d(
    W: np.ndarray,
    b: np.ndarray,
    target_vec_fn,
    k: int,
    quad: QuadratureConfig,
    reg_lambda: float,
) -> AssemblyResult:
    """
    3D antisymmetric-pair basis per neuron i:

        Phi_{i,12}(x) = s_i(x) * ( w2_i, -w1_i,  0 )
        Phi_{i,13}(x) = s_i(x) * ( w3_i,  0 ,  -w1_i)
        Phi_{i,23}(x) = s_i(x) * (  0 ,  w3_i, -w2_i)

    Unknowns are (a12, a13, a23) in R^M each => dof = 3M.

    We assemble:
        K_{ij} = ∫ s_i s_j
        u_r,i  = ∫ s_i v_r   (r=1..3)

    Then block Gram entries:
        A^{pq,rs}_{ij} = K_{ij} * (c_{i,pq} · c_{j,rs})
    and rhs:
        b^{pq}_i = c_{i,pq} · u_i  (with u_i = (u1_i,u2_i,u3_i)).
    """
    M = W.shape[0]
    K = np.zeros((M, M), dtype=np.float64)
    u1 = np.zeros((M,), dtype=np.float64)
    u2 = np.zeros((M,), dtype=np.float64)
    u3 = np.zeros((M,), dtype=np.float64)

    w1 = W[:, 0].astype(np.float64)
    w2 = W[:, 1].astype(np.float64)
    w3 = W[:, 2].astype(np.float64)

    for pts, wts in quad.iter(d=3):
        z = _linear_combo(pts, W, b)                      # (n,M)
        S = relu_k_derivative_scale(z, k)                 # (n,M)
        sqrtw = np.sqrt(wts)[:, None]
        S_w = S * sqrtw
        K += S_w.T @ S_w

        v = np.asarray(target_vec_fn(pts), dtype=np.float64)  # (n,3)
        u1 += S.T @ (wts * v[:, 0])
        u2 += S.T @ (wts * v[:, 1])
        u3 += S.T @ (wts * v[:, 2])

    dof = 3 * M
    A = np.zeros((dof, dof), dtype=np.float64)
    rhs = np.zeros((dof,), dtype=np.float64)

    # blocks indices
    i12 = slice(0, M)
    i13 = slice(M, 2 * M)
    i23 = slice(2 * M, 3 * M)

    # Fill diagonal blocks
    # 12-12: w2w2 + w1w1
    A[i12, i12] = K * (np.outer(w2, w2) + np.outer(w1, w1))
    # 13-13: w3w3 + w1w1
    A[i13, i13] = K * (np.outer(w3, w3) + np.outer(w1, w1))
    # 23-23: w3w3 + w2w2
    A[i23, i23] = K * (np.outer(w3, w3) + np.outer(w2, w2))

    # Off-diagonal blocks
    # 12-13: w2_i w3_j
    B_12_13 = K * np.outer(w2, w3)
    A[i12, i13] = B_12_13
    A[i13, i12] = B_12_13.T

    # 12-23: - w1_i w3_j
    B_12_23 = -K * np.outer(w1, w3)
    A[i12, i23] = B_12_23
    A[i23, i12] = B_12_23.T

    # 13-23: w1_i w2_j
    B_13_23 = K * np.outer(w1, w2)
    A[i13, i23] = B_13_23
    A[i23, i13] = B_13_23.T

    # RHS
    # b12_i = w2_i*u1_i + (-w1_i)*u2_i
    rhs[i12] = w2 * u1 - w1 * u2
    # b13_i = w3_i*u1_i + (-w1_i)*u3_i
    rhs[i13] = w3 * u1 - w1 * u3
    # b23_i = w3_i*u2_i + (-w2_i)*u3_i
    rhs[i23] = w3 * u2 - w2 * u3

    if reg_lambda and reg_lambda > 0:
        A = A + float(reg_lambda) * np.eye(dof)

    meta = {"dof": int(dof), "M": int(M)}
    return AssemblyResult(A=A, b=rhs, meta=meta)
