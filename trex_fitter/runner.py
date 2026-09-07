#!/usr/bin/env python3
"""Run the locally built, pinned TRExFitter executable."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_BINARY = ROOT / "source" / "build" / "bin" / "trex-fitter"


def executable() -> Path:
    """Return the source-build executable, honoring TREX_FITTER_BIN."""
    candidate = Path(os.environ.get("TREX_FITTER_BIN", DEFAULT_BINARY)).expanduser()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(
            f"TRExFitter executable not found at {candidate}. Build trex_fitter/source "
            "or set TREX_FITTER_BIN to an executable source build."
        )
    return candidate.resolve()


def run_action(binary: Path, action: str, config: Path, log_dir: Path) -> int:
    command = [str(binary), action, str(config)]
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{action}.stdout.log").write_text(completed.stdout)
    (log_dir / f"{action}.stderr.log").write_text(completed.stderr)
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, help="TRExFitter .config file")
    parser.add_argument("--actions", nargs="+", default=["n", "w", "f", "s"])
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "trex_fitter",
        help="Directory for per-action stdout and stderr logs.",
    )
    parser.add_argument("--check", action="store_true", help="Check that the source build is available.")
    args = parser.parse_args()

    try:
        binary = executable()
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if args.check:
        print(binary)
        return 0

    if args.config is None:
        parser.error("config is required unless --check is used")
    config = args.config.resolve()
    if config.suffix != ".config" or not config.is_file():
        parser.error(f"Expected an existing .config file, got {config}")

    for action in args.actions:
        result = run_action(binary, action, config, args.log_dir.resolve())
        if result:
            print(f"TRExFitter action {action!r} failed with exit code {result}", file=sys.stderr)
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
