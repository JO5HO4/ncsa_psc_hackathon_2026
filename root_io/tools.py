#!/usr/bin/env python3
"""Safe, read-only ROOT inspection tools with JSON-serializable results.

The functions in this module are the semantic tools used by ROOT-I/O dataset
episodes.  They deliberately expose a small interface instead of accepting
Python snippets or ROOT commands from a model.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
from pathlib import Path
from typing import Any


MAX_OBJECTS = 500
MAX_BRANCHES = 500
MAX_ENTRIES = 10_000
MAX_SAMPLE_VALUES = 10
MAX_PATTERNS = 20
MAX_NAME_LENGTH = 256


class RootInspectionError(RuntimeError):
    """A user-facing error raised for an invalid or unreadable ROOT request."""


def _load_dependencies():
    try:
        import awkward as ak
        import numpy as np
        import uproot
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise RootInspectionError(
            "ROOT inspection requires uproot, awkward, and numpy."
        ) from exc
    return uproot, ak, np


def _open_root(uproot: Any, path: Path):
    # Synchronous executors keep tool behavior predictable on batch systems and
    # avoid inheriting site-specific distributed-executor configuration. These
    # tools only accept local files, so bypass fsspec and its site plugins too.
    executor = uproot.source.futures.TrivialExecutor()
    return uproot.open(
        path,
        handler=uproot.source.file.MemmapSource,
        decompression_executor=executor,
        interpretation_executor=executor,
    )


def _resolve_root_file(path: str | Path, allowed_root: str | Path) -> tuple[Path, Path]:
    root = Path(allowed_root).resolve()
    requested = Path(path)
    if requested.is_absolute():
        raise RootInspectionError("ROOT file paths must be relative to the allowed root.")

    resolved = (root / requested).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RootInspectionError("ROOT file path escapes the allowed root.") from exc

    if resolved.suffix.lower() != ".root":
        raise RootInspectionError("Expected a file whose name ends in .root.")
    if not resolved.is_file():
        raise RootInspectionError(f"ROOT file does not exist: {relative.as_posix()}")
    return resolved, relative


def _bounded(value: int, *, name: str, minimum: int, maximum: int) -> int:
    if not minimum <= value <= maximum:
        raise RootInspectionError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _validated_name(value: str, *, name: str) -> str:
    if not value or len(value) > MAX_NAME_LENGTH or "\x00" in value:
        raise RootInspectionError(
            f"{name} must contain 1 to {MAX_NAME_LENGTH} non-NUL characters."
        )
    return value


def list_root_objects(
    path: str | Path,
    *,
    allowed_root: str | Path = ".",
    recursive: bool = True,
    max_objects: int = 100,
) -> dict[str, Any]:
    """List ROOT keys and basic metadata without reading event payloads."""

    max_objects = _bounded(max_objects, name="max_objects", minimum=1, maximum=MAX_OBJECTS)
    resolved, relative = _resolve_root_file(path, allowed_root)
    uproot, _, _ = _load_dependencies()

    objects: list[dict[str, Any]] = []
    with _open_root(uproot, resolved) as root_file:
        classnames = root_file.classnames(recursive=recursive, cycle=False)
        total = len(classnames)
        for name, class_name in list(classnames.items())[:max_objects]:
            item: dict[str, Any] = {"name": name, "class_name": class_name}
            try:
                obj = root_file[name]
                if hasattr(obj, "num_entries"):
                    item["entries"] = int(obj.num_entries)
                if hasattr(obj, "axes"):
                    item["dimensions"] = len(obj.axes)
            except Exception as exc:  # metadata failure should not hide inventory
                item["metadata_error"] = f"{type(exc).__name__}: {exc}"
            objects.append(item)

    return {
        "file": relative.as_posix(),
        "recursive": recursive,
        "object_count": total,
        "returned_object_count": len(objects),
        "truncated": total > len(objects),
        "objects": objects,
    }


def describe_tree(
    path: str | Path,
    tree: str,
    *,
    allowed_root: str | Path = ".",
    patterns: list[str] | None = None,
    max_branches: int = 200,
) -> dict[str, Any]:
    """Describe a TTree/RNTuple and its branches, optionally filtered by glob."""

    max_branches = _bounded(max_branches, name="max_branches", minimum=1, maximum=MAX_BRANCHES)
    resolved, relative = _resolve_root_file(path, allowed_root)
    uproot, _, _ = _load_dependencies()
    tree = _validated_name(tree, name="tree")
    selected_patterns = patterns or ["*"]
    if len(selected_patterns) > MAX_PATTERNS:
        raise RootInspectionError(f"patterns may contain at most {MAX_PATTERNS} items.")
    selected_patterns = [
        _validated_name(pattern, name="pattern") for pattern in selected_patterns
    ]

    with _open_root(uproot, resolved) as root_file:
        if tree not in root_file:
            raise RootInspectionError(f"Object not found: {tree}")
        tree_obj = root_file[tree]
        if not hasattr(tree_obj, "num_entries") or not hasattr(tree_obj, "typenames"):
            raise RootInspectionError(f"Object is not tree-like: {tree}")

        typenames = tree_obj.typenames()
        selected_names = [
            name
            for name in tree_obj.keys()
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in selected_patterns)
        ]
        branches = []
        for name in selected_names[:max_branches]:
            branch = tree_obj[name]
            root_type = typenames.get(name, getattr(branch, "typename", "unknown"))
            branches.append(
                {
                    "name": name,
                    "root_type": str(root_type),
                    "interpretation": str(getattr(branch, "interpretation", "unknown")),
                }
            )

        return {
            "file": relative.as_posix(),
            "tree": tree,
            "entries": int(tree_obj.num_entries),
            "patterns": selected_patterns,
            "matching_branch_count": len(selected_names),
            "returned_branch_count": len(branches),
            "truncated": len(selected_names) > len(branches),
            "branches": branches,
        }


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def summarize_branch(
    path: str | Path,
    tree: str,
    branch: str,
    *,
    allowed_root: str | Path = ".",
    max_entries: int = 1_000,
    max_sample_values: int = 5,
) -> dict[str, Any]:
    """Read a bounded prefix and return simple statistics for one branch."""

    max_entries = _bounded(max_entries, name="max_entries", minimum=1, maximum=MAX_ENTRIES)
    max_sample_values = _bounded(
        max_sample_values,
        name="max_sample_values",
        minimum=0,
        maximum=MAX_SAMPLE_VALUES,
    )
    tree = _validated_name(tree, name="tree")
    branch = _validated_name(branch, name="branch")
    resolved, relative = _resolve_root_file(path, allowed_root)
    uproot, ak, np = _load_dependencies()

    with _open_root(uproot, resolved) as root_file:
        if tree not in root_file:
            raise RootInspectionError(f"Object not found: {tree}")
        tree_obj = root_file[tree]
        if branch not in tree_obj:
            raise RootInspectionError(f"Branch not found in {tree}: {branch}")
        branch_obj = tree_obj[branch]
        entries_read = min(int(tree_obj.num_entries), max_entries)
        values = branch_obj.array(entry_stop=entries_read, library="ak")

    try:
        flattened = ak.flatten(values, axis=None)
    except Exception:
        flattened = values
    flattened = ak.drop_none(flattened)
    python_values = ak.to_list(flattened)
    if not isinstance(python_values, list):
        python_values = [python_values]

    result: dict[str, Any] = {
        "file": relative.as_posix(),
        "tree": tree,
        "branch": branch,
        "root_type": str(getattr(branch_obj, "typename", "unknown")),
        "entries_read": entries_read,
        "value_count": len(python_values),
        "sample_values": [_json_scalar(value) for value in python_values[:max_sample_values]],
    }

    if python_values:
        array = np.asarray(python_values)
        if np.issubdtype(array.dtype, np.bool_):
            result["true_fraction"] = float(np.mean(array))
        elif np.issubdtype(array.dtype, np.number):
            finite = array[np.isfinite(array)]
            result["finite_value_count"] = int(finite.size)
            if finite.size:
                result["minimum"] = float(np.min(finite))
                result["maximum"] = float(np.max(finite))
                result["mean"] = float(np.mean(finite))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowed-root", type=Path, default=Path("."))
    subparsers = parser.add_subparsers(dest="command", required=True)

    objects = subparsers.add_parser("list-objects")
    objects.add_argument("path")
    objects.add_argument("--no-recursive", action="store_true")
    objects.add_argument("--max-objects", type=int, default=100)

    tree = subparsers.add_parser("describe-tree")
    tree.add_argument("path")
    tree.add_argument("tree")
    tree.add_argument("--pattern", action="append", dest="patterns")
    tree.add_argument("--max-branches", type=int, default=200)

    branch = subparsers.add_parser("summarize-branch")
    branch.add_argument("path")
    branch.add_argument("tree")
    branch.add_argument("branch")
    branch.add_argument("--max-entries", type=int, default=1_000)
    branch.add_argument("--max-sample-values", type=int, default=5)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    common = {"allowed_root": args.allowed_root}
    try:
        if args.command == "list-objects":
            result = list_root_objects(
                args.path,
                recursive=not args.no_recursive,
                max_objects=args.max_objects,
                **common,
            )
        elif args.command == "describe-tree":
            result = describe_tree(
                args.path,
                args.tree,
                patterns=args.patterns,
                max_branches=args.max_branches,
                **common,
            )
        else:
            result = summarize_branch(
                args.path,
                args.tree,
                args.branch,
                max_entries=args.max_entries,
                max_sample_values=args.max_sample_values,
                **common,
            )
    except RootInspectionError as exc:
        print(json.dumps({"error": str(exc)}))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
