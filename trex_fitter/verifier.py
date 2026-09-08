#!/usr/bin/env python3
"""Cheap structural preflight for a TRExFitter configuration.

This deliberately does not attempt to model the entire TRExFitter language.
The upstream executable remains the authority for semantic validation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


BLOCK = re.compile(r'^(Job|MultiFit|Fit|Region|Sample|Systematic|NormFactor):\s*(?:"([^"]+)"|([^\s#]+))\s*(?:#.*)?$')
BLOCK_PREFIX = re.compile(r"^(Job|MultiFit|Fit|Region|Sample|Systematic|NormFactor):")
FIELD = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*):\s*(\S.*)?$")


@dataclass(frozen=True)
class Diagnostic:
    level: str
    code: str
    message: str
    line: int | None = None


def preflight(path: Path) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return [Diagnostic("error", "encoding", "Config must be UTF-8 text.")]

    blocks: dict[str, set[str]] = {}
    job_count = 0
    multifit_count = 0
    read_from = False
    for number, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        block = BLOCK.match(line)
        if block:
            kind, quoted_name, bare_name = block.groups()
            name = quoted_name or bare_name
            names = blocks.setdefault(kind, set())
            if kind in {"Job", "MultiFit"} and name in names:
                diagnostics.append(Diagnostic("error", "duplicate-block", f'Duplicate {kind} block "{name}".', number))
            names.add(name)
            job_count += kind == "Job"
            multifit_count += kind == "MultiFit"
            continue
        if BLOCK_PREFIX.match(line):
            diagnostics.append(Diagnostic("error", "malformed-block", "Block must have a name.", number))
            continue
        field = FIELD.match(line)
        if field and field.group(1) == "ReadFrom":
            read_from = True

    if not lines:
        diagnostics.append(Diagnostic("error", "empty", "Config is empty."))
    if job_count == 0 and multifit_count == 0:
        diagnostics.append(Diagnostic("error", "missing-job", "No Job or MultiFit block was found."))
    elif job_count > 1:
        diagnostics.append(Diagnostic("error", "multiple-jobs", "More than one Job block was found."))
    if job_count and not read_from:
        diagnostics.append(Diagnostic("warning", "missing-readfrom", "No ReadFrom field was found; verify the input mode."))
    if job_count and not blocks.get("Region"):
        diagnostics.append(Diagnostic("warning", "missing-region", "No Region blocks were found."))
    if job_count and not blocks.get("Sample"):
        diagnostics.append(Diagnostic("warning", "missing-sample", "No Sample blocks were found."))
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="TRExFitter .config file")
    parser.add_argument("--json", action="store_true", help="Emit diagnostics as JSON.")
    args = parser.parse_args()

    path = args.config.resolve()
    diagnostics: list[Diagnostic]
    if path.suffix != ".config" or not path.is_file():
        diagnostics = [Diagnostic("error", "missing-config", f"Expected an existing .config file, got {path}.")]
    else:
        diagnostics = preflight(path)

    valid = not any(item.level == "error" for item in diagnostics)
    report = {"config": str(path), "valid": valid, "diagnostics": [asdict(item) for item in diagnostics]}
    if args.json:
        print(json.dumps(report, indent=2))
    elif diagnostics:
        for item in diagnostics:
            location = f" line {item.line}" if item.line else ""
            print(f"{item.level.upper()}{location} [{item.code}]: {item.message}")
    else:
        print(f"VALID: {path}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
