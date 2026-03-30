"""Checkpoint IO utilities for lid-driven cavity experiments.

Responsibilities:
- Save/load FEM and FNS arrays.
- Save/load metadata for cache reuse checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


class CheckpointManager:
    """Read/write FEM/FNS checkpoint bundles (`.npz` + `.json`)."""

    def save_fem(self, path: Path, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Path:
        """Save FEM arrays and metadata.

        Typical shapes:
        - x:(Nx,), y:(Ny,), u:(Ny,Nx,2), grad:(Ny,Nx,2,2)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        return path

    def save_fns(self, path: Path, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Path:
        """Save FNS arrays and metadata.

        Typical shapes:
        - W:(M,2), b:(M,), a:(M,)
        """
        return self.save_fem(path, arrays, meta)

    def load(self, path: Path) -> Dict[str, Any]:
        """Load checkpoint arrays+metadata into a unified dictionary."""
        path = Path(path)
        arr = np.load(path, allow_pickle=False)
        meta = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
        return {"path": str(path), "meta": meta, "arrays": {k: arr[k] for k in arr.files}}

    @staticmethod
    def metadata_matches(path: Path, expected: Dict[str, Any]) -> bool:
        """Return True when checkpoint metadata contains exact expected key-values."""
        meta_path = Path(path).with_suffix(".json")
        if not Path(path).exists() or not meta_path.exists():
            return False
        got = json.loads(meta_path.read_text())
        for key, value in expected.items():
            if got.get(key) != value:
                return False
        return True
