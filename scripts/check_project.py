"""Run the same lightweight checks used by CI.

Usage:
    python scripts/check_project.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(label: str, command: list[str]) -> None:
    print(f"\n== {label} ==")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    _run(
        "compile",
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            ".",
            "-x",
            r"(\.git|\.venv|venv|data|sessions)",
        ],
    )
    _run("ruff", [sys.executable, "-m", "ruff", "check", "."])
    _run("tests", [sys.executable, "-m", "unittest", "discover", "-v"])
    print("\nAll project checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
