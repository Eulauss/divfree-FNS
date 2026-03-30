from __future__ import annotations

import sys

from divfree.tasks.stokes_lid.fem_vs_fns_runner import main


def _enable_live_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True, write_through=True)


if __name__ == "__main__":
    _enable_live_output()
    main()
