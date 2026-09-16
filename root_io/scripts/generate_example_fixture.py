#!/usr/bin/env python3
"""Generate the tiny local ROOT fixture used by documentation examples."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from root_io.fixtures import create_example_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/root_io/generated/diphoton-example.root",
    )
    args = parser.parse_args()
    output = create_example_fixture(args.output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"created={output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
