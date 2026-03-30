from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


def piecewise_gq_generator(d: int, Nx: int, order: int, chunk_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Composite (piecewise) tensor-product Gauss–Legendre quadrature on [-1,1]^d.

    Splits each axis into Nx cells, and uses order-point Gauss–Legendre per axis
    within each cell. Yields (points, weights) in chunks.

    points: (n_chunk, d), weights: (n_chunk,)
    """
    if d < 1:
        raise ValueError("d must be >= 1")
    Nx = int(Nx); order = int(order); chunk_size = int(chunk_size)
    if Nx <= 0 or order <= 0 or chunk_size <= 0:
        raise ValueError("Nx, order, chunk_size must be positive integers")

    x_1d, w_1d = np.polynomial.legendre.leggauss(order)

    # Base points/weights for one cell centered at 0: [-h/2, h/2]^d
    mesh = np.meshgrid(*([x_1d] * d), indexing="ij")
    gauss_pts_base = np.array([m.ravel() for m in mesh]).T  # (order^d, d)

    mesh_w = np.meshgrid(*([w_1d] * d), indexing="ij")
    weights_base = np.prod([m.ravel() for m in mesh_w], axis=0)  # (order^d,)

    h = 2.0 / Nx
    scale = h / 2.0
    gauss_pts_base = gauss_pts_base * scale
    weights_base = weights_base * (scale ** d)

    # cell centers / translations
    idx = np.arange(0, Nx)
    centers_mesh = np.meshgrid(*([idx] * d), indexing="ij")
    cell_centers = np.array([m.ravel() for m in centers_mesh]).T  # (Nx^d, d)
    cell_translations = -1.0 + h / 2.0 + cell_centers * h         # (Nx^d, d)

    n_per_cell = order ** d
    num_cells = Nx ** d
    num_total_pts = n_per_cell * num_cells

    start_idx = 0
    while start_idx < num_total_pts:
        end_idx = min(start_idx + chunk_size, num_total_pts)

        cell_start = start_idx // n_per_cell
        cell_end = (end_idx - 1) // n_per_cell

        pts_list = []
        w_list = []

        for cell_idx in range(cell_start, cell_end + 1):
            translation = cell_translations[cell_idx]

            local_start = 0 if cell_idx > cell_start else start_idx % n_per_cell
            local_end = n_per_cell if cell_idx < cell_end else (end_idx - 1) % n_per_cell + 1

            pts_list.append(gauss_pts_base[local_start:local_end] + translation)
            w_list.append(weights_base[local_start:local_end])

        yield np.vstack(pts_list), np.concatenate(w_list)
        start_idx = end_idx


@dataclass(frozen=True)
class QuadratureConfig:
    Nx: int = 12
    order: int = 3
    chunk_size: int = 5000

    def iter(self, d: int):
        return piecewise_gq_generator(d=d, Nx=self.Nx, order=self.order, chunk_size=self.chunk_size)


def volume_of_cube(d: int) -> float:
    return float(2.0 ** d)
