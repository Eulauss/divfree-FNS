from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from divfree.tasks.l2_approx.targets import TargetBase


def _bubble(t: np.ndarray) -> np.ndarray:
    return (1.0 - t * t) ** 2


def _bubble_prime(t: np.ndarray) -> np.ndarray:
    return -4.0 * t * (1.0 - t * t)


def _bubble_second(t: np.ndarray) -> np.ndarray:
    return -4.0 + 12.0 * t * t


@dataclass
class StokesTarget2D(TargetBase):
    """
    2D divergence-free zero-Dirichlet target from a stream function.
    psi(x, y) = b(x)b(y)sin(kx)sin(ky)
    u = (d_y psi, - d_x psi)
    where b is bubble function, k is the frequency.
    """
    freq: float = np.pi

    def dim(self) -> int:
        return 2

    def eval(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        X = x[:, 0]
        Y = x[:, 1]
        f = float(self.freq)

        bx = _bubble(X)
        by = _bubble(Y)
        bx_p = _bubble_prime(X)
        by_p = _bubble_prime(Y)

        sx = np.sin(f * X)
        cx = np.cos(f * X)
        sy = np.sin(f * Y)
        cy = np.cos(f * Y)

        u1 = bx * (by_p * sx * sy + by * sx * f * cy)
        u2 = -by * (bx_p * sx * sy + bx * f * cx * sy)
        return np.stack([u1, u2], axis=1)

    def eval_grad(self, x: np.ndarray) -> np.ndarray:
        """
        Return grad (jacobian) u with shape (n,2,2), where grad u[r, j] = d_{x_j} u_r.
        """
        x = np.asarray(x, dtype=np.float64)
        X = x[:, 0]
        Y = x[:, 1]
        f = float(self.freq)

        bx = _bubble(X)
        by = _bubble(Y)
        bx_p = _bubble_prime(X)
        by_p = _bubble_prime(Y)
        bx_pp = _bubble_second(X)
        by_pp = _bubble_second(Y)

        sx = np.sin(f * X)
        cx = np.cos(f * X)
        sy = np.sin(f * Y)
        cy = np.cos(f * Y)

        # second derivatives of the stream function
        dxy = (
            bx_p * by_p * sx * sy
            + bx * by_p * f * cx * sy
            + bx_p * by * sx * f * cy
            + bx * by * f * f * cx * cy
        )
        dxx = (
            bx_pp * by * sx * sy
            + 2.0 * bx_p * by * f * cx * sy
            - bx * by * f * f * sx * sy
        )
        dyy = (
            bx * by_pp * sx * sy
            + 2.0 * bx * by_p * sx * f * cy
            - bx * by * f * f * sx * sy
        )

        grad = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
        # u1 = d_y psi
        grad[:, 0, 0] = dxy
        grad[:, 0, 1] = dyy
        # u2 = - d_x psi
        grad[:, 1, 0] = -dxx
        grad[:, 1, 1] = -dxy
        return grad


@dataclass
class StokesTarget3D(TargetBase):
    """
    3D divergence-free zero-Dirichlet target from a vector potential:
    0      mu12    mu13
    -mu12  0       mu23
    -mu13  -mu23   0
    where 
        mu12 = \beta(\mathbf{x}) \sin(kx) \sin(ky)
        mu13 = -\beta(\mathbf{x}) \sin(kx) \sin(kz)
        mu23 = \beta(\mathbf{x}) \sin(ky) \sin(kz),

        beta(\mathbf{x}) = b(x)b(y)b(z)
    then the dic-free vector field is
        u1 = d_y mu12 + d_z mu13
        u2 = d_x -mu12 + d_z mu23
        u3 = d_x -mu13 + d_y -mu23
    """
    freq: float = np.pi

    def dim(self) -> int:
        return 3

    def eval(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        X = x[:, 0]
        Y = x[:, 1]
        Z = x[:, 2]
        f = float(self.freq)

        bx = _bubble(X)
        by = _bubble(Y)
        bz = _bubble(Z)

        bx_p = _bubble_prime(X)
        by_p = _bubble_prime(Y)
        bz_p = _bubble_prime(Z)

        beta = bx * by * bz
        dx_beta = bx_p * by * bz
        dy_beta = bx * by_p * bz
        dz_beta = bx * by * bz_p

        sx = np.sin(f * X)
        sy = np.sin(f * Y)
        sz = np.sin(f * Z)
        cx = np.cos(f * X)
        cy = np.cos(f * Y)
        cz = np.cos(f * Z)

        term_x = dx_beta * sx + beta * f * cx
        term_y = dy_beta * sy + beta * f * cy
        term_z = dz_beta * sz + beta * f * cz

        u1 = sx * (term_y - term_z)
        u2 = sy * (term_z - term_x)
        u3 = sz * (term_x - term_y)
        return np.stack([u1, u2, u3], axis=1)

    def eval_grad(self, x: np.ndarray) -> np.ndarray:
        """
        Return grad(jacobian) u with shape (n,3,3), where grad u[r, j] = d_{x_j} u_r.
        """
        x = np.asarray(x, dtype=np.float64)
        X = x[:, 0]
        Y = x[:, 1]
        Z = x[:, 2]
        f = float(self.freq)

        bx = _bubble(X)
        by = _bubble(Y)
        bz = _bubble(Z)
        bx_p = _bubble_prime(X)
        by_p = _bubble_prime(Y)
        bz_p = _bubble_prime(Z)
        bx_pp = _bubble_second(X)
        by_pp = _bubble_second(Y)
        bz_pp = _bubble_second(Z)

        beta = bx * by * bz
        dx_beta = bx_p * by * bz
        dy_beta = bx * by_p * bz
        dz_beta = bx * by * bz_p

        dxx = bx_pp * by * bz
        dyy = bx * by_pp * bz
        dzz = bx * by * bz_pp
        dxy = bx_p * by_p * bz
        dxz = bx_p * by * bz_p
        dyz = bx * by_p * bz_p

        sx = np.sin(f * X)
        sy = np.sin(f * Y)
        sz = np.sin(f * Z)
        cx = np.cos(f * X)
        cy = np.cos(f * Y)
        cz = np.cos(f * Z)

        term_x = dx_beta * sx + beta * f * cx
        term_y = dy_beta * sy + beta * f * cy
        term_z = dz_beta * sz + beta * f * cz

        dterm_x_dx = dxx * sx + 2.0 * dx_beta * f * cx - beta * f * f * sx
        dterm_x_dy = dxy * sx + dy_beta * f * cx
        dterm_x_dz = dxz * sx + dz_beta * f * cx

        dterm_y_dx = dxy * sy + dx_beta * f * cy
        dterm_y_dy = dyy * sy + 2.0 * dy_beta * f * cy - beta * f * f * sy
        dterm_y_dz = dyz * sy + dz_beta * f * cy

        dterm_z_dx = dxz * sz + dx_beta * f * cz
        dterm_z_dy = dyz * sz + dy_beta * f * cz
        dterm_z_dz = dzz * sz + 2.0 * dz_beta * f * cz - beta * f * f * sz

        grad = np.zeros((x.shape[0], 3, 3), dtype=np.float64)

        # u1 = sx * (term_y - term_z)
        grad[:, 0, 0] = f * cx * (term_y - term_z) + sx * (dterm_y_dx - dterm_z_dx)
        grad[:, 0, 1] = sx * (dterm_y_dy - dterm_z_dy)
        grad[:, 0, 2] = sx * (dterm_y_dz - dterm_z_dz)

        # u2 = sy * (term_z - term_x)
        grad[:, 1, 0] = sy * (dterm_z_dx - dterm_x_dx)
        grad[:, 1, 1] = f * cy * (term_z - term_x) + sy * (dterm_z_dy - dterm_x_dy)
        grad[:, 1, 2] = sy * (dterm_z_dz - dterm_x_dz)

        # u3 = sz * (term_x - term_y)
        grad[:, 2, 0] = sz * (dterm_x_dx - dterm_y_dx)
        grad[:, 2, 1] = sz * (dterm_x_dy - dterm_y_dy)
        grad[:, 2, 2] = f * cz * (term_x - term_y) + sz * (dterm_x_dz - dterm_y_dz)

        return grad


def build_target(problem_type: str, target_cfg: Dict) -> TargetBase:
    if problem_type == "stokes2d":
        return StokesTarget2D(**target_cfg)
    if problem_type == "stokes3d":
        return StokesTarget3D(**target_cfg)
    raise ValueError(f"Unknown problem.type: {problem_type}")
