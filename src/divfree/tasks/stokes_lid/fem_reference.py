"""Exact FEM reference solve/load utilities for 2D lid-driven Stokes.

This module provides:
- `FEMReferenceSolver2D`: solves Stokes with Taylor-Hood (P_p/P_{p-1}) and saves
  exact velocity DOFs only (no structured-grid sampling / no finite-difference gradient).
- `ExactFEMEvaluator`: reconstructs the FEM function from saved DOFs and evaluates
  both velocity and exact polynomial gradient at arbitrary points via dolfinx eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .checkpointing import CheckpointManager
from .lid_bc import build_lid_bc


def _imports():
    """Import dolfinx stack lazily and raise a clear error if unavailable."""
    try:
        from mpi4py import MPI
        from petsc4py import PETSc
        import ufl
        import basix
        from basix.ufl import element, mixed_element
        from dolfinx import fem, geometry, mesh
        from dolfinx.fem.petsc import LinearProblem
        return MPI, PETSc, ufl, basix, element, mixed_element, fem, geometry, mesh, LinearProblem
    except Exception as exc:
        raise RuntimeError(
            "dolfinx/mpi4py/petsc4py/ufl/basix are required for exact FEM reference/evaluation."
        ) from exc


@dataclass
class ExactFEMEvaluator:
    """Rebuild exact FEM velocity function from checkpoint and evaluate on points."""

    dofs_array: np.ndarray
    mesh_N: int
    fem_order: int

    def __post_init__(self):
        MPI, PETSc, ufl, basix, element, _mixed_element, fem, geometry, mesh, _LinearProblem = _imports()
        self._MPI = MPI
        self._ufl = ufl
        self._fem = fem
        self._geometry = geometry
        self._mesh = mesh
        self._basix = basix

        comm = MPI.COMM_WORLD
        if comm.size != 1:
            raise RuntimeError("ExactFEMEvaluator currently supports single-rank MPI only.")

        # Reconstruct mesh and function space exactly from checkpoint metadata.
        self.domain = mesh.create_rectangle(
            comm,
            [np.array([-1.0, -1.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
            [int(self.mesh_N), int(self.mesh_N)],
            cell_type=mesh.CellType.triangle,
        )

        V_el = element("Lagrange", self.domain.basix_cell(), int(self.fem_order), shape=(2,))
        self.V = fem.functionspace(self.domain, V_el)

        self.uh = fem.Function(self.V)
        if self.uh.x.array.shape[0] != np.asarray(self.dofs_array).shape[0]:
            raise RuntimeError(
                f"DOF length mismatch: ckpt={np.asarray(self.dofs_array).shape[0]} vs space={self.uh.x.array.shape[0]}"
            )
        self.uh.x.array[:] = np.asarray(self.dofs_array, dtype=np.float64)

        # Build exact DG representation of grad(u): each component is polynomial of order p-1 per cell.
        DG_tensor = basix.ufl.element("DG", self.domain.basix_cell(), int(self.fem_order) - 1, shape=(2, 2))
        self.W_grad = fem.functionspace(self.domain, DG_tensor)
        self.grad_uh = fem.Function(self.W_grad)
        grad_expr = fem.Expression(ufl.grad(self.uh), self.W_grad.element.interpolation_points())
        self.grad_uh.interpolate(grad_expr)

    def _find_cells(self, pts3: np.ndarray) -> np.ndarray:
        """Find containing cell for each query point; raise if any point is outside mesh."""
        bb_tree = self._geometry.bb_tree(self.domain, self.domain.topology.dim)
        candidates = self._geometry.compute_collisions_points(bb_tree, pts3)
        colliding = self._geometry.compute_colliding_cells(self.domain, candidates, pts3)

        cells = np.full((pts3.shape[0],), -1, dtype=np.int32)
        for i in range(pts3.shape[0]):
            links = colliding.links(i)
            if len(links) > 0:
                cells[i] = int(links[0])
        if np.any(cells < 0):
            bad = int(np.sum(cells < 0))
            raise RuntimeError(f"{bad} evaluation points are outside FEM mesh domain.")
        return cells

    def evaluate_at_points(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate exact velocity and gradient at points.

        Args:
            pts: (N,2)
        Returns:
            u_exact: (N,2)
            grad_exact: (N,2,2)
        """
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Expected pts shape (N,2), got {pts.shape}")

        pts3 = np.zeros((pts.shape[0], 3), dtype=np.float64)
        pts3[:, :2] = pts
        cells = self._find_cells(pts3)

        u_val = self.uh.eval(pts3, cells)
        g_val = self.grad_uh.eval(pts3, cells)
        return np.asarray(u_val, dtype=np.float64), np.asarray(g_val, dtype=np.float64).reshape((-1, 2, 2))


@dataclass
class FEMReferenceSolver2D:
    """Solve FEM reference and cache exact velocity DOFs."""

    ckpt: CheckpointManager

    def _solve_taylor_hood(self, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Solve Stokes on [-1,1]^2 using Taylor-Hood P_p/P_{p-1}."""
        MPI, PETSc, ufl, _basix, element, mixed_element, fem, _geometry, mesh, LinearProblem = _imports()

        comm = MPI.COMM_WORLD
        if comm.size != 1:
            raise RuntimeError("Current FEM solve/checkpoint supports single-rank runs only.")

        ref_cfg = config.get("reference", {})
        N_mesh = int(ref_cfg.get("fem_mesh_N", 64))
        p = int(ref_cfg.get("fem_order", 2))
        if p < 2:
            raise ValueError("fem_order must be >= 2 for Taylor-Hood P_p/P_{p-1}.")
        nu = float(config.get("nu", 1.0))

        domain = mesh.create_rectangle(
            comm,
            [np.array([-1.0, -1.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
            [N_mesh, N_mesh],
            cell_type=mesh.CellType.triangle,
        )

        V_el = element("Lagrange", domain.basix_cell(), p, shape=(2,))
        Q_el = element("Lagrange", domain.basix_cell(), p - 1)
        W = fem.functionspace(domain, mixed_element([V_el, Q_el]))

        (u, p_trial) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)
        f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))

        a = (
            nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.inner(p_trial, ufl.div(v)) * ufl.dx
            - ufl.inner(ufl.div(u), q) * ufl.dx
        )
        L = ufl.inner(f, v) * ufl.dx

        fdim = domain.topology.dim - 1
        top_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
        wall_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.isclose(x[1], -1.0) | np.isclose(x[0], -1.0) | np.isclose(x[0], 1.0)
        )

        V_u, _ = W.sub(0).collapse()
        lid_bc = build_lid_bc(config.get("lid", {}))

        u_top = fem.Function(V_u)

        def top_interp(x):
            pts = np.stack([x[0], x[1]], axis=1)
            vals = lid_bc.eval_on_boundary(pts)
            return np.vstack([vals[:, 0], vals[:, 1]])

        u_top.interpolate(top_interp)
        dofs_top = fem.locate_dofs_topological((W.sub(0), V_u), fdim, top_facets)
        bc_top = fem.dirichletbc(u_top, dofs_top, W.sub(0))

        u_noslip = fem.Function(V_u)
        u_noslip.x.array[:] = 0.0
        dofs_walls = fem.locate_dofs_topological((W.sub(0), V_u), fdim, wall_facets)
        bc_walls = fem.dirichletbc(u_noslip, dofs_walls, W.sub(0))

        # --- Task 1: Pin pressure to remove nullspace ---
        Q, _ = W.sub(1).collapse()
        pin_facets = mesh.locate_entities_boundary(
            domain, 0, lambda x: np.isclose(x[0], -1.0) & np.isclose(x[1], -1.0)
        )
        dofs_pin = fem.locate_dofs_topological((W.sub(1), Q), 0, pin_facets)
        p_pin = fem.Function(Q)
        p_pin.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p_pin, dofs_pin, W.sub(1))

        # Solve with the extra pressure constraint
        wh = LinearProblem(
            a,
            L,
            bcs=[bc_top, bc_walls, bc_p],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        ).solve()
        uh_collapsed = wh.sub(0).collapse()

        # --- Task 2: Safe standalone interpolation for DOF alignment ---
        V_el_standalone = element("Lagrange", domain.basix_cell(), p, shape=(2,))
        V_standalone = fem.functionspace(domain, V_el_standalone)
        uh_safe = fem.Function(V_standalone)
        uh_safe.interpolate(uh_collapsed)

        return {
            "dofs_array": np.asarray(uh_safe.x.array, dtype=np.float64),
            "mesh_N": np.asarray([N_mesh], dtype=np.int32),
            "fem_order": np.asarray([p], dtype=np.int32),
        }

    def solve(self, config: Dict[str, Any]) -> Path:
        out_dir = Path(config["output_dir"])
        ref_cfg = config.get("reference", {})
        path = Path(ref_cfg.get("checkpoint_path") or (out_dir / "reference" / "fem_reference_exact.npz"))

        meta_key = {
            "method": "fem",
            "fem_scheme": "dolfinx_taylor_hood_exact_dofs",
            "nu": float(config.get("nu", 1.0)),
            "domain": config.get("domain", "[-1,1]^2"),
            "lid": config.get("lid", {}),
            "fem_mesh_N": int(ref_cfg.get("fem_mesh_N", 64)),
            "fem_order": int(ref_cfg.get("fem_order", 2)),
        }
        if bool(ref_cfg.get("reuse_if_exists", True)) and self.ckpt.metadata_matches(path, meta_key):
            return path

        arrays = self._solve_taylor_hood(config)
        return self.ckpt.save_fem(path, arrays=arrays, meta=meta_key)
