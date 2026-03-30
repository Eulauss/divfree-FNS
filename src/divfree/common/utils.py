from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import numpy as np


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))


def jsonable(x: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, default=jsonable))


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=jsonable) + "\n")


@dataclass
class Timer:
    t0: float = 0.0

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    @property
    def dt(self) -> float:
        return time.perf_counter() - self.t0
