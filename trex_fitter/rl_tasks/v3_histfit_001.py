#!/usr/bin/env python3
"""Prepare and evaluate the first native TRExFitter RL task.

The agent edits only ``analysis.config`` in an isolated workspace.  The
evaluator first awards a small structural reward, then optionally runs the
pinned native TRExFitter ``hwfs`` chain and scores observable fit outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from trex_fitter.config_format import ConfigError, _parse_blocks, _unquote


PROJECT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parent / "fixtures" / "v3-histfit-001"
TASK_FILE = TASK_DIR / "TASK.md"
TEMPLATE = TASK_DIR / "analysis.config"
BASELINE_EXPECTED_SIGNIFICANCE = 0.497381


def workspace_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT / path
    path = path.resolve()
    try:
        path.relative_to(PROJECT)
    except ValueError as error:
        raise ValueError("workspace must be inside the repository") from error
    return path


def prepare(workspace: Path) -> dict[str, object]:
    workspace.mkdir(parents=True, exist_ok=True)
    config = workspace / "analysis.config"
    task = workspace / "TASK.md"
    if config.exists() or task.exists():
        raise FileExistsError(f"refusing to overwrite existing task files in {workspace}")
    shutil.copy2(TEMPLATE, config)
    shutil.copy2(TASK_FILE, task)
    return {"workspace": str(workspace), "config": str(config), "task": str(task)}


def structural_checks(config: Path) -> tuple[dict[str, bool], str | None, str | None]:
    """Cheap, native-HIST checks; the full native run remains authoritative."""
    checks = {
        "config_exists": config.is_file(), "job": False, "fit": False,
        "region": False, "data": False, "background": False,
        "signal": False, "norm_factor": False,
    }
    if not config.is_file():
        return checks, None, "analysis.config is missing"
    try:
        blocks = _parse_blocks(config)
    except (ConfigError, UnicodeDecodeError) as error:
        return checks, None, str(error)

    by_kind: dict[str, list] = {}
    for block in blocks:
        by_kind.setdefault(block.kind, []).append(block)
    job_name = None
    if len(by_kind.get("Job", [])) == 1:
        job = by_kind["Job"][0]
        job_name = job.name
        checks["job"] = (
            _unquote(job.values.get("ReadFrom", "")).upper() == "HIST"
            and bool(_unquote(job.values.get("POI", "")))
            and bool(_unquote(job.values.get("HistoPath", "")))
        )
    if len(by_kind.get("Fit", [])) >= 1:
        checks["fit"] = all(
            _unquote(block.values.get("FitType", "")).upper() == "SPLUSB"
            and _unquote(block.values.get("FitRegion", "")).upper() == "CRSR"
            for block in by_kind["Fit"]
        )
    checks["region"] = any(bool(_unquote(block.values.get("HistoName", ""))) for block in by_kind.get("Region", []))
    samples = by_kind.get("Sample", [])
    checks["data"] = any(_unquote(block.values.get("Type", "")).upper() == "DATA" and bool(_unquote(block.values.get("HistoFile", ""))) for block in samples)
    checks["background"] = any(_unquote(block.values.get("Type", "")).upper() == "BACKGROUND" and bool(_unquote(block.values.get("HistoFile", ""))) for block in samples)
    checks["signal"] = any(_unquote(block.values.get("Type", "")).upper() == "SIGNAL" and bool(_unquote(block.values.get("HistoFile", ""))) for block in samples)
    # TRExFitter's compact HIST syntax puts the constrained signal
    # normalization directly on the Sample block.  A standalone NormFactor
    # block would reject the known-good FitExample reference configuration.
    checks["norm_factor"] = any(
        _unquote(block.values.get("Type", "")).upper() == "SIGNAL"
        and bool(_unquote(block.values.get("NormFactor", "")))
        for block in samples
    )
    return checks, job_name, None


def run_native(config: Path, workspace: Path, timeout: int) -> tuple[dict[str, object] | None, str | None]:
    result_dir = workspace / "results"
    run_report = result_dir / "run.json"
    command = [
        sys.executable, "-m", "trex_fitter.runner", str(config.relative_to(PROJECT)),
        "--backend", "trex", "--actions", "h", "w", "f", "s",
        "--output-dir", str((result_dir / "output").relative_to(PROJECT)),
        "--log-dir", str((result_dir / "logs").relative_to(PROJECT)),
        "--json", str(run_report.relative_to(PROJECT)), "--timeout", str(timeout),
    ]
    completed = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True, check=False)
    if not run_report.is_file():
        return None, f"runner did not write {run_report}; exit={completed.returncode}; stderr={completed.stderr[-1000:]}"
    report = json.loads(run_report.read_text(encoding="utf-8"))
    report["evaluator_runner_exit_code"] = completed.returncode
    return report, None


def evaluate(workspace: Path, execute: bool, timeout: int) -> dict[str, object]:
    config = workspace / "analysis.config"
    checks, job_name, parse_error = structural_checks(config)
    structural_fraction = sum(checks.values()) / len(checks)
    reward = round(0.20 * structural_fraction, 6)
    report: dict[str, object] = {
        "task_id": "v3-histfit-001", "workspace": str(workspace), "config": str(config),
        "checks": checks, "job_name": job_name, "parse_error": parse_error,
        "executed": execute, "reward": reward, "success": False,
    }
    if not execute or not all(checks.values()) or job_name is None:
        report["reason"] = "structural checks incomplete" if not all(checks.values()) else "execution not requested"
        return report

    native, error = run_native(config, workspace, timeout)
    report["native_run"] = native
    if error:
        report["reason"] = error
        return report
    if not native.get("success"):
        report["reason"] = str(native.get("error", "native TRExFitter run failed"))
        return report
    reward += 0.40

    output = workspace / "results" / "output" / job_name
    artifacts = {
        "histograms": output / "Histograms" / f"{job_name}_histos.root",
        "fit": output / "Fits" / f"{job_name}.root",
        "workspace": output / "RooStats" / f"{job_name}_combined_{job_name}_model.root",
    }
    artifact_checks = {name: path.is_file() for name, path in artifacts.items()}
    report["artifact_checks"] = artifact_checks
    if all(artifact_checks.values()):
        reward += 0.15

    log_dir = Path(str(native["log_dir"]))
    # The runner forwards TRExFitter diagnostics to the container stderr log
    # on Perlmutter, while some local backends use stdout. Read both without
    # treating runner supervision logs as fit diagnostics.
    native_logs = list(log_dir.glob("trex-fitter_*.stdout.log")) + list(log_dir.glob("trex-fitter_*.stderr.log"))
    log_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in native_logs)
    healthy_fit = "Number of bad fits: 0" in log_text
    report["healthy_fit"] = healthy_fit
    if healthy_fit:
        reward += 0.10

    significance = native.get("significance")
    finite_significance = isinstance(significance, (int, float)) and math.isfinite(significance)
    report["finite_expected_significance"] = finite_significance
    if finite_significance and significance > 0:
        reward += 0.05
    reaches_baseline = finite_significance and significance >= 0.95 * BASELINE_EXPECTED_SIGNIFICANCE
    report["reaches_reference_baseline"] = reaches_baseline
    if reaches_baseline:
        reward += 0.10

    report["reward"] = round(reward, 6)
    report["success"] = all(artifact_checks.values()) and healthy_fit and reaches_baseline
    report["reason"] = "success" if report["success"] else "native run completed but terminal quality checks failed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "evaluate"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--workspace", required=True)
    evaluator = subparsers.choices["evaluate"]
    evaluator.add_argument("--execute", action="store_true", help="run native TRExFitter hwfs after structural checks")
    evaluator.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    workspace = workspace_path(args.workspace)
    if args.command == "prepare":
        print(json.dumps(prepare(workspace), indent=2))
        return
    print(json.dumps(evaluate(workspace, args.execute, args.timeout), indent=2))


if __name__ == "__main__":
    main()
