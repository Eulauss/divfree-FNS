"""Boundary condition models for 2D lid-driven cavity experiments.

This module defines an OOP interface (`LidBCBase`) and concrete lid profiles used by
`stokes_lid.fem_reference` and `stokes_lid.fns_solver`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LidBCBase:
    """Base interface for cavity boundary velocity g(x)."""

    def eval_on_boundary(self, points: np.ndarray) -> np.ndarray:
        """Evaluate boundary velocity.

        Shapes:
        - points: (N,2)
        - return g: (N,2)
        """
        raise NotImplementedError


@dataclass
class ClassicLidBC(LidBCBase):
    """Classic cavity BC: top=(U,0), others=(0,0)."""

    U: float = 1.0
    tol: float = 1e-12

    def eval_on_boundary(self, points: np.ndarray) -> np.ndarray:
        """Evaluate piecewise-constant lid BC on boundary points."""
        pts = np.asarray(points, dtype=np.float64)  # (N,2)
        g = np.zeros((pts.shape[0], 2), dtype=np.float64)  # (N,2)
        top = np.isclose(pts[:, 1], 1.0, atol=self.tol)  # (N,)
        g[top, 0] = float(self.U)
        return g


@dataclass
class SmoothLidBC(LidBCBase):
    """Smooth lid BC with corner-vanishing tangential profile."""

    U: float = 1.0
    profile: str = "poly4"
    tol: float = 1e-12

    def _top_profile(self, x: np.ndarray) -> np.ndarray:
        """Evaluate top-wall speed profile at x-coordinates.

        Shape:
        - x: (N,)
        - return: (N,)
        """
        # Domain mapping: repository domain is [-1,1], profile is defined on [0,1].
        s = 0.5 * (x + 1.0)
        if self.profile == "poly4":
            # Scale by 16 so max value equals U (unscaled max is U/16 at s=0.5).
            return float(self.U) * 16.0 * (s**2) * ((1.0 - s) ** 2)
        if self.profile == "cos2":
            # Smooth alternative profile on [0,1]: sin^2(pi s).
            return float(self.U) * np.sin(np.pi * s) ** 2
        raise ValueError(f"Unknown smooth profile: {self.profile}")

    def eval_on_boundary(self, points: np.ndarray) -> np.ndarray:
        """Evaluate smooth-lid BC values on boundary points."""
        pts = np.asarray(points, dtype=np.float64)  # (N,2)
        g = np.zeros((pts.shape[0], 2), dtype=np.float64)  # (N,2)
        top = np.isclose(pts[:, 1], 1.0, atol=self.tol)  # (N,)
        g[top, 0] = self._top_profile(pts[top, 0])
        return g


def build_lid_bc(cfg: dict) -> LidBCBase:
    """Factory for lid BC implementations from config dict."""
    lid_type = str(cfg.get("type", "classic")).lower()
    U = float(cfg.get("U", 1.0))
    if lid_type == "classic":
        return ClassicLidBC(U=U)
    if lid_type == "smooth":
        return SmoothLidBC(U=U, profile=str(cfg.get("smooth_profile", "poly4")))
    raise ValueError(f"Unsupported lid type: {lid_type}")
