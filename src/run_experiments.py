from __future__ import annotations

import json
import sys
from pathlib import Path

from divfree.tasks.l2_approx.runner import ExperimentRunner


def _enable_live_output() -> None:
    # Improve progress visibility under `conda run`/non-tty pipes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True, write_through=True)


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def main() -> None:
    _enable_live_output()
    if len(sys.argv) < 2:
        print("Usage: python src/run_experiments.py <config.json> [mass_matrix=true|false]")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())
    if len(sys.argv) >= 3:
        cfg.setdefault("experiment", {})["mass_matrix"] = _parse_bool(sys.argv[2])

    out = ExperimentRunner(cfg=cfg).run()
    print(f"\nDone. Output dir: {out}")


if __name__ == "__main__":
    main()
