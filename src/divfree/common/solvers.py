from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.linalg
import scipy.sparse.linalg


class SolverBase:
    name: str = "base"

    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError


@dataclass
class CholeskySolver(SolverBase):
    name: str = "cholesky"
    jitter: float = 0.0  # optional diagonal jitter if needed

    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        Aj = A
        if self.jitter and self.jitter > 0:
            Aj = Aj + float(self.jitter) * np.eye(A.shape[0])

        # cho_factor expects SPD; may raise LinAlgError
        c, lower = scipy.linalg.cho_factor(Aj, lower=True, check_finite=False)
        x = scipy.linalg.cho_solve((c, lower), b, check_finite=False)
        info = {"solver": self.name, "lower": bool(lower), "jitter": float(self.jitter)}
        return x, info


@dataclass
class LstsqSolver(SolverBase):
    """Robust least squares via SVD driver (gelsd by default)."""
    name: str = "lstsq"
    rcond: Optional[float] = None
    lapack_driver: str = "gelsd"
    save_svd: bool = False

    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        x, residuals, rank, s = scipy.linalg.lstsq(
            A, b, cond=self.rcond, lapack_driver=self.lapack_driver, check_finite=False
        )
        info = {
            "solver": self.name,
            "lapack_driver": self.lapack_driver,
            "rank": int(rank),
            "singular_values_min": float(np.min(s)) if s.size else None,
            "singular_values_max": float(np.max(s)) if s.size else None,
            "residuals_sum": float(np.sum(residuals)) if np.size(residuals) else None,
        }
        if self.save_svd:
            info["singular_values"] = np.sort(s).tolist()
        return x, info


@dataclass
class CGSolver(SolverBase):
    name: str = "cg"
    tol: float = 1e-10
    maxiter: int = 20000

    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        x, info_cg = scipy.sparse.linalg.cg(A, b, tol=float(self.tol), maxiter=int(self.maxiter))
        info = {"solver": self.name, "tol": float(self.tol), "maxiter": int(self.maxiter), "cg_info": int(info_cg)}
        return x, info


@dataclass
class LSQRSolver(SolverBase):
    name: str = "lsqr"
    damp: float = 0.0
    atol: float = 1e-10
    btol: float = 1e-10
    iter_lim: Optional[int] = None
    show: bool = False

    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        result = scipy.sparse.linalg.lsqr(
            A,
            b,
            damp=float(self.damp),
            atol=float(self.atol),
            btol=float(self.btol),
            iter_lim=self.iter_lim,
            show=bool(self.show),
        )
        x = result[0]
        info = {
            "solver": self.name,
            "damp": float(self.damp),
            "atol": float(self.atol),
            "btol": float(self.btol),
            "iter_lim": None if self.iter_lim is None else int(self.iter_lim),
            "show": bool(self.show),
            "istop": int(result[1]),
            "itn": int(result[2]),
            "r1norm": float(result[3])
        }
        return x, info


@dataclass
class QRSolver(SolverBase):
    """Least squares via QR decomposition for overdetermined systems."""
    name: str = "qr"

    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if A.shape[0] < A.shape[1]:
            raise ValueError("QR solver only supports overdetermined systems (rows >= cols)")
        q, r = scipy.linalg.qr(A, mode="economic")
        qt_b = q.T @ b
        x = scipy.linalg.solve_triangular(r, qt_b, lower=False, check_finite=False)
        info = {"solver": self.name, "rows": int(A.shape[0]), "cols": int(A.shape[1])}
        return x, info


@dataclass
class AutoSolver(SolverBase):
    """Heuristic choice: try Cholesky, else lstsq for small, else CG."""
    name: str = "auto"
    max_cholesky_dof: int = 8000
    max_lstsq_dof: int = 6000
    cholesky_jitter: float = 0.0
    cg_tol: float = 1e-10
    cg_maxiter: int = 20000

    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        dof = int(A.shape[0])

        if dof <= self.max_cholesky_dof:
            try:
                x, info = CholeskySolver(jitter=self.cholesky_jitter).solve(A, b)
                info["auto_choice"] = "cholesky"
                return x, info
            except Exception as e:
                # fall through
                pass

        if dof <= self.max_lstsq_dof:
            x, info = LstsqSolver().solve(A, b)
            info["auto_choice"] = "lstsq"
            return x, info

        x, info = CGSolver(tol=self.cg_tol, maxiter=self.cg_maxiter).solve(A, b)
        info["auto_choice"] = "cg"
        return x, info


def build_solver(cfg: Dict, reg_lambda: Optional[float] = None) -> SolverBase:
    name = (cfg or {}).get("name", "auto")
    name = str(name).lower()

    if name == "cholesky":
        return CholeskySolver(jitter=float((cfg.get("cholesky") or {}).get("jitter", 0.0)))
    if name == "lstsq":
        lcfg = cfg.get("lstsq") or {}
        return LstsqSolver(
            rcond=lcfg.get("rcond", None),
            lapack_driver=str(lcfg.get("lapack_driver", "gelsd")),
            save_svd=bool(lcfg.get("save_svd", False)),
        )
    if name == "cg":
        ccfg = cfg.get("cg") or {}
        return CGSolver(
            tol=float(ccfg.get("tol", 1e-10)),
            maxiter=int(ccfg.get("maxiter", 20000)),
        )
    if name == "lsqr":
        lcfg = cfg.get("lsqr") or {}
        damp = lcfg.get("damp", None)
        if damp is None and reg_lambda is not None and reg_lambda > 0:
            damp = float(np.sqrt(reg_lambda))
        return LSQRSolver(
            damp=float(damp or 0.0),
            atol=float(lcfg.get("atol", 1e-10)),
            btol=float(lcfg.get("btol", 1e-10)),
            iter_lim=lcfg.get("iter_lim", None),
            show=bool(lcfg.get("show", False)),
        )
    if name == "qr":
        return QRSolver()
    if name == "auto":
        acfg = cfg.get("auto") or {}
        ccfg = cfg.get("cg") or {}
        chcfg = cfg.get("cholesky") or {}
        return AutoSolver(
            max_cholesky_dof=int(acfg.get("max_cholesky_dof", 8000)),
            max_lstsq_dof=int(acfg.get("max_lstsq_dof", 6000)),
            cholesky_jitter=float(chcfg.get("jitter", 0.0)),
            cg_tol=float(ccfg.get("tol", 1e-10)),
            cg_maxiter=int(ccfg.get("maxiter", 20000)),
        )
    raise ValueError(f"Unknown solver name: {name}")
