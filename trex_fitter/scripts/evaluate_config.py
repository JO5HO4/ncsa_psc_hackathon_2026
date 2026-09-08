#!/usr/bin/env python3
"""Evaluate one completed TRExFitter config and emit a JSON result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from interface import TrexFitterInterface


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a completed TRExFitter .config file.")
    parser.add_argument("config", type=Path, help="Project-local .config file to evaluate.")
    parser.add_argument("--mock", action="store_true", help="Use the fast deterministic mock runner.")
    parser.add_argument("--actions", nargs="+", default=["n", "w", "f", "s"])
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    evaluator = (
        TrexFitterInterface.mock(project_dir=PROJECT_DIR, timeout=args.timeout)
        if args.mock
        else TrexFitterInterface(project_dir=PROJECT_DIR, timeout=args.timeout)
    )
    result = evaluator.run(args.config, actions=args.actions, log_dir=args.log_dir)
    print(json.dumps(result.to_dict(), indent=2))
    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
