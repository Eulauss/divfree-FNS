from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np


class TargetBase:
    def dim(self) -> int:
        raise NotImplementedError

    def input_dim(self) -> int:
        return self.dim()

    def eval(self, x: np.ndarray) -> np.ndarray:
        """x: (n, d) -> y: (n, dim) or (n,) for scalar."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ------------------ scalar targets ------------------

@dataclass
class ScalarTarget(TargetBase):
    d: int = 1
    kind: Literal["sin", "poly", "exp", "sin_tensor_product", "gaussian_kernel"] = "sin_tensor_product"
    freq: float = np.pi
    sigma: float = 1.0

    def dim(self) -> int:
        return 1

    def input_dim(self) -> int:
        return int(self.d)

    def _coerce_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError(f"Expected x to have shape (n, d); got {x.shape}")
        if x.shape[1] != int(self.d):
            raise ValueError(f"Expected input dimension {self.d}, got {x.shape[1]}")
        return x

    def eval(self, x: np.ndarray) -> np.ndarray:
        x = self._coerce_input(x)
        if self.kind in {"sin", "sin_tensor_product"}:
            return np.prod(np.sin(self.freq * x), axis=1)
        if self.kind == "gaussian_kernel":
            sigma = float(self.sigma)
            if sigma <= 0:
                raise ValueError("sigma must be positive")
            norm2 = np.sum(x * x, axis=1)
            return np.exp(-norm2 / (2.0 * sigma ** 2))
        if self.kind == "poly":
            if int(self.d) != 1:
                raise ValueError("poly target is only defined for d=1")
            x1 = x[:, 0]
            return 0.5 + x1 - 2.0 * x1**2 + 0.3 * x1**3
        if self.kind == "exp":
            if int(self.d) != 1:
                raise ValueError("exp target is only defined for d=1")
            x1 = x[:, 0]
            return np.exp(x1) - np.mean(np.exp(x1))
        raise ValueError(f"Unknown ScalarTarget kind: {self.kind}")


# ------------------ 2D div-free targets (v = curl psi) ------------------

@dataclass
class DivFreeTarget2D(TargetBase):
    kind: Literal["stream_sin"] = "stream_sin"
    freq: float = np.pi

    def dim(self) -> int:
        return 2

    def eval(self, x: np.ndarray) -> np.ndarray:
        # x: (n,2)
        x = np.asarray(x, dtype=np.float64)
        X = x[:, 0]; Y = x[:, 1]
        if self.kind == "stream_sin":
            # psi = sin(pi x) sin(pi y)
            # v = ( d_y psi, - d_x psi )
            f = self.freq
            dpsi_dx = f * np.cos(f * X) * np.sin(f * Y)
            dpsi_dy = f * np.sin(f * X) * np.cos(f * Y)
            return np.stack([dpsi_dy, -dpsi_dx], axis=1)
        raise ValueError(f"Unknown DivFreeTarget2D kind: {self.kind}")


# ------------------ 3D div-free targets (v = curl A) ------------------

@dataclass
class DivFreeTarget3D(TargetBase):
    kind: Literal["curl_sin"] = "curl_sin"
    freq: float = np.pi

    def dim(self) -> int:
        return 3

    def eval(self, x: np.ndarray) -> np.ndarray:
        # x: (n,3)
        x = np.asarray(x, dtype=np.float64)
        X = x[:, 0]; Y = x[:, 1]; Z = x[:, 2]
        f = self.freq

        if self.kind == "curl_sin":
            # Vector potential:
            # A = (sin(fY)sin(fZ), sin(fZ)sin(fX), sin(fX)sin(fY))
            # v = curl A:
            # v1 = dY A3 - dZ A2
            # v2 = dZ A1 - dX A3
            # v3 = dX A2 - dY A1
            A1 = np.sin(f * Y) * np.sin(f * Z)
            A2 = np.sin(f * Z) * np.sin(f * X)
            A3 = np.sin(f * X) * np.sin(f * Y)

            dY_A3 = f * np.sin(f * X) * np.cos(f * Y)
            dZ_A2 = f * np.cos(f * Z) * np.sin(f * X)

            dZ_A1 = f * np.sin(f * Y) * np.cos(f * Z)
            dX_A3 = f * np.cos(f * X) * np.sin(f * Y)

            dX_A2 = f * np.cos(f * X) * np.sin(f * Z)
            dY_A1 = f * np.cos(f * Y) * np.sin(f * Z)

            v1 = dY_A3 - dZ_A2
            v2 = dZ_A1 - dX_A3
            v3 = dX_A2 - dY_A1
            return np.stack([v1, v2, v3], axis=1)

        raise ValueError(f"Unknown DivFreeTarget3D kind: {self.kind}")


def build_target(problem_type: str, target_cfg: Dict) -> TargetBase:
    """Factory."""
    if problem_type in {"scalar1d", "scalar"}:
        return ScalarTarget(**target_cfg)
    if problem_type == "divfree2d":
        return DivFreeTarget2D(**target_cfg)
    if problem_type == "divfree3d":
        return DivFreeTarget3D(**target_cfg)
    raise ValueError(f"Unknown problem.type: {problem_type}")
