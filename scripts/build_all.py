#!/usr/bin/env python3
"""Rebuild all figures used in the paper.

Usage:
    poetry run python scripts/build_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    "scripts/simulate.py",
    "scripts/product_control.py",
    "scripts/branches.py",
    "scripts/one_reality.py",
    "scripts/spatial_multiscale.py",
    "scripts/scaling.py",
    "scripts/projectors.py",
    "scripts/robustness.py",
    "scripts/nesting_table.py",
]

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    failed = []
    t_total = time.time()

    for script in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"  {script}")
        print(f"{'='*60}")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, script],
            cwd=root,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            failed.append(script)
            print(f"  FAILED ({elapsed:.0f}s)")
        else:
            print(f"  OK ({elapsed:.0f}s)")

    print(f"\n{'='*60}")
    print(f"  Total: {time.time() - t_total:.0f}s")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("  All scripts completed successfully.")
