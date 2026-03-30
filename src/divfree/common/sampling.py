from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple, Optional, Dict
import hashlib

import numpy as np
from scipy.special import betainc, betaincinv

# ============================================================
# Geometry helpers
# ============================================================

def hyperplane_intersects_cube(w: np.ndarray, b: float, cube_min: float = -1.0, cube_max: float = 1.0) -> bool:
    """
    Check whether the hyperplane {x: w·x + b = 0} intersects the axis-aligned cube [cube_min,cube_max]^d.

    We compute extrema of dot(w,x) over the box using bounds (equivalent to checking corners).
    Intersects iff -b is within [min dot(w,x), max dot(w,x)].
    """
    w = np.asarray(w, dtype=np.float64)
    dot_max = float(np.sum(np.where(w >= 0, w * cube_max, w * cube_min)))
    dot_min = float(np.sum(np.where(w >= 0, w * cube_min, w * cube_max)))
    val = -float(b)
    return (dot_min <= val) and (val <= dot_max)


# ============================================================
# Sphere point generators
# ============================================================

def _gauss_normalize(n: int, dim: int) -> np.ndarray:
    x = np.random.randn(int(n), int(dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-15
    return x.astype(np.float64)


def _energy_points_jax(
    n_points: int,
    dim: int,
    *,
    max_iter: int = 80,
    jit: bool = True,
    seed: int = 0,
    dtype: str = "float64",
) -> np.ndarray:
    """
    Energy minimization on the sphere via JAX + jaxopt.LBFGS, following the logic in `get_uniform_sphere_points_jax`.
    (Riesz energy with exponent s = dim-1, using chordal distances.)

    Requires: jax, jaxopt. For GPU acceleration install the appropriate jaxlib build.
    """
    try:
        import jax
        from jax import config as jax_config
        import jax.numpy as jnp
        from jaxopt import LBFGS
    except Exception as e:
        raise ImportError(
            "Energy sampler requires `jax` and `jaxopt`. "
            "Install them (and the correct jaxlib for your CUDA) or use sample_alg='gauss_normalize'."
        ) from e

    n_points = int(n_points)
    dim = int(dim)
    max_iter = int(max_iter)
    seed = int(seed)

    if dtype == "float64":
        jax_config.update("jax_enable_x64", True)
        jdtype = jnp.float64
    else:
        jdtype = jnp.float32

    key = jax.random.PRNGKey(seed)
    eps = jnp.asarray(1e-12, dtype=jdtype)

    # init: Gaussian then normalize
    points = jax.random.normal(key, (n_points, dim), dtype=jdtype)
    norms0 = jnp.linalg.norm(points, axis=1, keepdims=True)
    points = points / jnp.maximum(norms0, eps)

    # precompute upper-triangle indices once (avoid double counting)
    pi_np, pj_np = np.triu_indices(n_points, k=1)
    pair_i = jnp.asarray(pi_np, dtype=jnp.int32)
    pair_j = jnp.asarray(pj_np, dtype=jnp.int32)

    s_exp = jnp.asarray(dim - 1.0, dtype=jdtype)  # Riesz exponent s = dim-1

    def project_to_sphere(P):
        nrm = jnp.linalg.norm(P, axis=1, keepdims=True)
        return P / jnp.maximum(nrm, eps)

    def potential_energy(X):
        diff = X[:, None, :] - X[None, :, :]
        dist2 = jnp.sum(diff * diff, axis=-1)
        r2 = dist2[pair_i, pair_j]
        r2 = jnp.maximum(r2, eps)
        return jnp.sum(r2 ** (-s_exp / 2.0))

    def objective(x_flat):
        X = x_flat.reshape(n_points, dim)
        X = project_to_sphere(X)
        return potential_energy(X)

    fun = jax.jit(objective) if jit else objective
    lbfgs = LBFGS(fun=fun, maxiter=max_iter, jit=jit)
    x_opt, _ = lbfgs.run(points.flatten())
    X_opt = project_to_sphere(x_opt.reshape(n_points, dim))

    return np.array(X_opt, dtype=np.float64)


def _random_rotation_matrix(dim: int) -> np.ndarray:
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive")
    if dim == 1:
        return np.eye(1, dtype=np.float64)
    A = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(A)
    diag = np.sign(np.diag(R))
    diag[diag == 0] = 1.0
    Q = Q * diag
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q.astype(np.float64)


def _cache_key_energy(n_points: int, dim: int, max_iter: int, jit: bool, seed: int, dtype: str) -> str:
    raw = f"energy|n={int(n_points)}|d={int(dim)}|it={int(max_iter)}|jit={int(bool(jit))}|seed={int(seed)}|dtype={dtype}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def sample_uniform_sphere(
    n: int,
    dim: int,
    *,
    alg: Literal["gauss_normalize", "energy"] = "gauss_normalize",
    energy_max_iter: int = 80,
    energy_jit: bool = True,
    energy_seed: int = 0,
    energy_dtype: str = "float64",
    cache_dir: Optional[str] = ".sphere_cache",
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Sample n points on S^{dim-1} in R^{dim}.

    - alg='gauss_normalize': Gaussian then normalize (fast).
    - alg='energy': Riesz-energy minimization using JAX (more uniform, slower).
      Uses on-disk cache by default (cache_dir) to avoid recomputation.

    Returns ndarray shape (n, dim), float64.
    """
    n = int(n); dim = int(dim)
    if n <= 0 or dim <= 0:
        raise ValueError("n and dim must be positive")

    if alg == "gauss_normalize":
        return _gauss_normalize(n, dim)

    if alg != "energy":
        raise ValueError(f"Unknown alg: {alg}")

    # energy: cache
    cdir = Path(cache_dir) if cache_dir else None
    cache_hit = False
    if cdir is not None:
        cdir.mkdir(parents=True, exist_ok=True)
        key = _cache_key_energy(n, dim, energy_max_iter, energy_jit, energy_seed, energy_dtype)
        path = cdir / f"sphere_{key}.npz"
        if path.exists() and (not force_recompute):
            try:
                data = np.load(path, allow_pickle=False)
                pts = data["points"]
                print(f"Reuse cached sampling inner weights of shape {pts.shape}")
                if isinstance(pts, np.ndarray):
                    cache_hit = True
                    return pts.astype(np.float64)
            except Exception:
                pass

    if not cache_hit:
        try:
            import jax

            devices = jax.devices()
            if devices:
                device_desc = ", ".join(str(d) for d in devices)
                print(f"JAX devices: {device_desc}")
        except Exception:
            pass

    pts = _energy_points_jax(
        n_points=n,
        dim=dim,
        max_iter=energy_max_iter,
        jit=energy_jit,
        seed=energy_seed,
        dtype=energy_dtype,
    )

    if cdir is not None:
        try:
            np.savez_compressed(path, points=pts)
        except Exception:
            pass

    return pts


# ============================================================
# Neuron sampler
# ============================================================

@dataclass
class NeuronSampler:
    d: int
    param_space: Literal["Sd", "SxB"] = "Sd"
    positive_bias: bool = False
    intersection_filter: bool = True
    # b_range: Tuple[float, float] = (-1.0, 1.0)
    max_tries_factor: int = 50

    # how to sample sphere points
    sample_alg: Literal["gauss_normalize", "energy"] = "gauss_normalize"
    random_rotate_sphere: bool = False
    interval_bias_sampling: Literal["uniform", "density_transform"] = "uniform"

    # Energy-sampler knobs (used only if sample_alg='energy')
    energy_max_iter: int = 80
    energy_jit: bool = True
    energy_seed: int = 0
    energy_dtype: str = "float64"
    energy_cache_dir: Optional[str] = ".sphere_cache"
    energy_force_recompute: bool = False

    def _sphere(self, n: int, dim: int) -> np.ndarray:
        '''
        sample quasi-uniform M_init points on the Sphere
        before filtering
        '''
        return sample_uniform_sphere(
            n=n,
            dim=dim,
            alg=self.sample_alg,
            energy_max_iter=self.energy_max_iter,
            energy_jit=self.energy_jit,
            energy_seed=self.energy_seed,
            energy_dtype=self.energy_dtype,
            cache_dir=self.energy_cache_dir,
            force_recompute=self.energy_force_recompute,
        )

    def sample(self, M: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            W: (M, d)
            b: (M,)
        """
        M = int(M)
        if M <= 0:
            raise ValueError("M must be positive")
        d = int(self.d)
        # lo, hi = map(float, self.b_range)

        first_energy = (self.sample_alg == "energy")

        if first_energy:
            n_prop = int(M)
            if self.param_space == "Sd":
                theta = self._sphere(n_prop, d + 1)  # (n_prop, d+1)
                W = theta[:, :d]
                b = theta[:, d]

            # apply filters once
            if self.positive_bias:
                mask = b > 0
                W = W[mask]
                b = b[mask]

            if self.intersection_filter:
                keep = [i for i in range(W.shape[0]) if hyperplane_intersects_cube(W[i], float(b[i]))]
                W = W[keep] if keep else W[:0]
                b = b[keep] if keep else b[:0]

        return W, b
