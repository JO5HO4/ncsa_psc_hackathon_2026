#!/usr/bin/env python3
"""Run, inspect, and compare ROOT SFT experiments."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data/datasets/atlas-open-data-sft-dataset"
ARTIFACT_ROOT = REPO_ROOT / "artifacts/atlas-open-data-sft"
RUNS_ROOT = ARTIFACT_ROOT / "runs"
REGISTRY_PATH = ARTIFACT_ROOT / "runs.json"
LOCK_PATH = ARTIFACT_ROOT / ".runs.lock"
MODEL_PROFILES = {
    "qwen3.5-0.8b": "0.8b",
    "qwen3.5-9b": "9b",
}
RESOLVED_MODELS = {
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
}
ELIGIBLE_PHASES = {"validated", "inferred", "scored", "plotted", "ready"}
RUN_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*\Z")


def now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    os.replace(temp, path)


@contextlib.contextmanager
def registry_lock() -> Iterator[None]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError:
            # Lustre mounts exposed through Podman-HPC can reject flock with
            # EREMOTEIO. mkdir is atomic on the shared filesystem and keeps
            # host and container writers coordinated in that case.
            lock_dir = LOCK_PATH.with_suffix(".lockdir")
            deadline = time.monotonic() + 60
            while True:
                try:
                    lock_dir.mkdir()
                    break
                except FileExistsError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Another root-sft command is still updating the run registry.")
                    time.sleep(0.1)
            try:
                yield
            finally:
                lock_dir.rmdir()
            return
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"version": 1, "runs": {}}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def git_revision(path: Path) -> str | None:
    completed = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_path(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def stored_path(path: Path) -> str:
    """Store paths relative to the checkout so host and container agree."""
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def saved_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def report_python(requested: str | None, *, dry_run: bool) -> str:
    """Return a Python interpreter that can build the Parquet score report."""
    candidates = (
        requested,
        os.environ.get("ROOT_SFT_REPORT_PYTHON"),
        str(REPO_ROOT / ".venv/bin/python"),
        sys.executable,
    )
    if dry_run:
        return next(candidate for candidate in candidates if candidate)
    tried: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in tried:
            continue
        tried.append(candidate)
        try:
            probe = subprocess.run(
                [candidate, "-c", "import pyarrow.parquet"],
                capture_output=True,
            )
        except OSError:
            continue
        if probe.returncode == 0:
            return candidate
    raise RuntimeError(
        "A working report Python with PyArrow is required. Set "
        "ROOT_SFT_REPORT_PYTHON before entering the ROOT container, or pass "
        "--report-python PATH. Tried: " + ", ".join(tried)
    )


def validate_run_name(value: str) -> str:
    if value == "latest" or not RUN_NAME.fullmatch(value):
        raise ValueError("Run names use letters, numbers, dots, dashes, and underscores; 'latest' is reserved.")
    return value


def generated_run_id(model: str) -> str:
    return f"{model.replace('.', '')}-{dt.datetime.now(tz=dt.timezone.utc):%Y%m%d-%H%M%S}"


def dataset_inputs() -> dict[str, Path]:
    return {
        "train": DATASET_ROOT / "data/sft/train.parquet",
        "validation": DATASET_ROOT / "data/sft/validation.parquet",
        "test": DATASET_ROOT / "data/sft/test.parquet",
    }


def dataset_metadata() -> dict[str, Any]:
    files = dataset_inputs()
    return {
        "path": stored_path(DATASET_ROOT),
        "revision": git_revision(DATASET_ROOT),
        "inputs": {name: {"path": stored_path(path), "sha256": sha256(path)} for name, path in files.items()},
    }


def create_run(run_id: str, *, model: str, seed: int | None, command: list[str], kind: str) -> dict[str, Any]:
    validate_run_name(run_id)
    directory = run_path(run_id)
    with registry_lock():
        registry = read_registry()
        if run_id in registry["runs"] or directory.exists():
            raise FileExistsError(f"Run '{run_id}' already exists. Use './root-sft resume {run_id}' to continue it.")
        for name in ("checkpoint", "prompts", "completions", "benchmark", "plots", "logs"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "id": run_id,
            "kind": kind,
            "model": model,
            "model_revision": {"requested": model, "resolved": RESOLVED_MODELS.get(model, model)},
            "seed": seed,
            "phase": "created",
            "eligible": False,
            "created_at": now(),
            "updated_at": now(),
            "command": command,
            "repository_revision": git_revision(REPO_ROOT),
            "verl_revision": git_revision(REPO_ROOT / "verl"),
            "runtime": {"python": sys.version.split()[0]},
            "dataset": dataset_metadata(),
            "paths": {
                "run": stored_path(directory), "checkpoint": stored_path(directory / "checkpoint"),
                "prompts": stored_path(directory / "prompts"), "completions": stored_path(directory / "completions"),
                "benchmark": stored_path(directory / "benchmark"), "plots": stored_path(directory / "plots"),
                "logs": stored_path(directory / "logs"),
            },
            "phases": [{"phase": "created", "at": now()}],
        }
        atomic_json(directory / "run.json", record)
        registry["runs"][run_id] = {"path": stored_path(directory / "run.json"), "phase": "created", "eligible": False, "updated_at": record["updated_at"], "model": model}
        atomic_json(REGISTRY_PATH, registry)
    return record


def load_run(run_spec: str, *, require_eligible: bool = False) -> dict[str, Any]:
    with registry_lock():
        registry = read_registry()
        if run_spec == "latest":
            choices = [(item["updated_at"], name) for name, item in registry["runs"].items() if item.get("eligible")]
            if not choices:
                raise ValueError("There is no validated run yet. Train a run or use 'infer --model MODEL --run NAME' first.")
            run_spec = max(choices)[1]
        item = registry["runs"].get(run_spec)
        if item is None:
            raise ValueError(f"Run '{run_spec}' was not found. Use './root-sft runs' to see available runs.")
        record = json.loads(saved_path(item["path"]).read_text(encoding="utf-8"))
    if require_eligible and not record.get("eligible"):
        raise ValueError(f"Run '{record['id']}' is not ready yet (current step: {record['phase']}).")
    return record


def update_run(record: dict[str, Any], phase: str, *, eligible: bool | None = None, **extra: Any) -> dict[str, Any]:
    record = dict(record)
    record.update(extra)
    record["phase"] = phase
    if eligible is not None:
        record["eligible"] = eligible
    record["updated_at"] = now()
    record.setdefault("phases", []).append({"phase": phase, "at": record["updated_at"]})
    with registry_lock():
        atomic_json(saved_path(record["paths"]["run"]) / "run.json", record)
        registry = read_registry()
        registry["runs"][record["id"]] = {"path": stored_path(saved_path(record["paths"]["run"]) / "run.json"), "phase": phase, "eligible": record["eligible"], "updated_at": record["updated_at"], "model": record["model"]}
        atomic_json(REGISTRY_PATH, registry)
    return record


def fail_run(record: dict[str, Any], phase: str, code: int, log: Path, message: str) -> None:
    update_run(record, "failed", eligible=False, failure={"phase": phase, "exit_code": code, "log": str(log), "message": message})


def clear_failed_outputs(record: dict[str, Any]) -> dict[str, Any]:
    """Clear only a failed run's generated outputs after explicit --force."""
    directory = saved_path(record["paths"]["run"]).resolve()
    expected = run_path(record["id"]).resolve()
    if directory != expected:
        raise RuntimeError("Refusing to clear a run whose saved path does not match its run name.")
    for name in ("checkpoint", "prompts", "completions", "benchmark", "plots", "logs"):
        target = saved_path(record["paths"][name]).resolve()
        if not target.is_relative_to(directory):
            raise RuntimeError("Refusing to clear output outside the selected run directory.")
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
    record.pop("failure", None)
    record.pop("completions", None)
    record.pop("report", None)
    record.pop("summary", None)
    record.pop("plots", None)
    return update_run(record, "created", eligible=False, reset_at=now())


def in_verl_container() -> bool:
    return os.environ.get("ROOT_SFT_IN_CONTAINER") == "1" or (Path("/workspace").is_dir() and (Path("/workspace") / "verl").is_dir())


def require_verl_container() -> None:
    if not in_verl_container():
        raise RuntimeError("This command needs the GPU training container. Start it with 'bash training/scripts/container.sh', then run this command from /workspace.")


def command_log(record: dict[str, Any], name: str) -> Path:
    return saved_path(record["paths"]["logs"]) / f"{name}.log"


def execute(command: list[str], *, log: Path, cwd: Path | None = None, timeout: int | None = None, env: dict[str, str] | None = None, dry_run: bool = False) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log.write_text("DRY RUN\n" + " ".join(command) + "\n", encoding="utf-8")
        return 0
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            return process.wait(timeout=timeout)
        except BaseException:
            process.kill()
            process.wait()
            raise


def profile_for(model: str) -> str:
    if model not in MODEL_PROFILES:
        raise ValueError("Choose one of: qwen3.5-0.8b, qwen3.5-9b.")
    return MODEL_PROFILES[model]


def cmd_check(args: argparse.Namespace) -> int:
    checks: list[tuple[str, str, str]] = []
    checks.append(("repository", "PASS" if (REPO_ROOT / ".git").exists() else "FAIL", str(REPO_ROOT)))
    files = dataset_inputs()
    checks.append(("dataset", "PASS" if all(path.is_file() for path in files.values()) else "FAIL", "ROOT SFT data files"))
    try:
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        probe = ARTIFACT_ROOT / ".write-check"
        probe.touch(); probe.unlink()
        checks.append(("workspace", "PASS", "run output location is writable"))
    except OSError:
        checks.append(("workspace", "FAIL", "cannot write run output location"))
    checks.append(("GPU container", "PASS" if in_verl_container() else "WARN", "start with: bash training/scripts/container.sh"))
    checks.append(("GPU", "PASS" if shutil.which("nvidia-smi") else "WARN", "needed only for training and inference"))
    checks.append(("inference runtime", "PASS" if (REPO_ROOT / "verl").is_dir() else "FAIL", "initialize the repository dependencies"))
    checks.append(("ROOT", "PASS" if shutil.which("root") else "WARN", "for scoring: source /global/cfs/cdirs/atlas/scripts/setupATLAS.sh; setupATLAS -c centos7+batch; lsetup \"root 6.30.02-x86_64-centos7-gcc11-opt\""))
    checks.append(("ROOT fallback", "PASS" if (REPO_ROOT / "training/scripts/root_runtime.sh").is_file() else "FAIL", "CVMFS fallback launcher"))
    checks.append(("score sandbox", "PASS" if shutil.which("bwrap") else "FAIL", "install bubblewrap before scoring generated commands"))
    checks.append(("report Python", "PASS" if shutil.which("python3") else "FAIL", "needed for reports and plots"))
    free_gib = shutil.disk_usage(ARTIFACT_ROOT).free / 1024**3
    checks.append(("free space", "PASS" if free_gib >= 20 else "WARN", f"{free_gib:.1f} GiB available for runs"))
    for package, label in (("pyarrow", "report data"), ("matplotlib", "plots")):
        code = subprocess.run([sys.executable, "-c", f"import {package}"], capture_output=True).returncode
        checks.append((label, "PASS" if code == 0 else "WARN", f"install the reporting dependencies for {label}"))
    for label, result, detail in checks:
        print(f"{result:<4} {label}: {detail}")
    return 1 if any(result == "FAIL" for _, result, _ in checks) else 0


def cmd_train(args: argparse.Namespace) -> int:
    model = args.model
    profile = profile_for(model)
    learning_rate = getattr(args, "learning_rate", 1e-5)
    batch_size = getattr(args, "batch_size", 16)
    lora_rank = getattr(args, "lora_rank", 16)
    lora_alpha = getattr(args, "lora_alpha", 16)
    run_id = args.run or generated_run_id(model)
    if args.force and args.resume:
        raise ValueError("Choose either --resume or --force, not both.")
    if args.force:
        if not args.run:
            raise ValueError("Choose the failed run to restart with --run NAME --force.")
        record = load_run(args.run)
        if record["phase"] != "failed":
            raise ValueError("--force only clears outputs from a failed run.")
        record = clear_failed_outputs(record)
    elif args.resume:
        if not args.run:
            raise ValueError("Choose the failed run to resume with --run NAME --resume.")
        record = load_run(args.run)
        if record["phase"] != "failed":
            raise ValueError(f"Run '{record['id']}' has not failed and does not need resume.")
        if record["model"] != model:
            raise ValueError("A resumed run must keep its original model.")
    else:
        record = create_run(run_id, model=model, seed=args.seed, command=sys.argv[1:], kind="trained")
    log = command_log(record, "train")
    try:
        require_verl_container()
        inputs = dataset_inputs()
        if any("test.parquet" in str(path) for path in (inputs["train"], inputs["validation"])):
            raise ValueError("Training may only use the train and validation splits.")
        record = update_run(record, "training", training={"epochs": args.epochs, "profile": profile, "learning_rate": learning_rate, "batch_size": batch_size, "lora_rank": lora_rank, "lora_alpha": lora_alpha, "save_frequency": "after_each_epoch", "test_frequency": "after_each_epoch", "train_file": stored_path(inputs["train"]), "validation_file": stored_path(inputs["validation"])})
        env = os.environ.copy()
        env.update({"QWEN35_MODEL_SIZE": profile, "TOTAL_EPOCHS": str(args.epochs), "LR": str(learning_rate), "TRAIN_BATCH_SIZE": str(batch_size), "LORA_RANK": str(lora_rank), "LORA_ALPHA": str(lora_alpha), "SAVE_FREQ": "after_each_epoch", "TEST_FREQ": "after_each_epoch", "TRAIN_FILE": str(inputs["train"]), "VAL_FILE": str(inputs["validation"]), "SAVE_DIR": str(saved_path(record["paths"]["checkpoint"])), "EXPERIMENT_NAME": run_id})
        code = execute(["bash", str(REPO_ROOT / "training/scripts/run_verl_sft.sh")], log=log, env=env, dry_run=args.dry_run)
        if code:
            tracker = saved_path(record["paths"]["checkpoint"]) / "latest_checkpointed_iteration.txt"
            if tracker.is_file() and tracker.read_text(encoding="utf-8").strip().isdigit():
                print("Training returned a nonzero status after saving a checkpoint; exporting that completed checkpoint.")
                return cmd_finalize(argparse.Namespace(run=record["id"], dry_run=args.dry_run))
            fail_run(record, "training", code, log, "Training stopped before an export was validated.")
            return code
        record = update_run(record, "checkpointed")
        record = update_run(record, "exported")
        tracker = saved_path(record["paths"]["checkpoint"]) / "latest_checkpointed_iteration.txt"
        inference_model = record["paths"]["checkpoint"]
        if tracker.is_file():
            step = tracker.read_text(encoding="utf-8").strip()
            if step.isdigit():
                inference_model = stored_path(saved_path(record["paths"]["checkpoint"]) / f"global_step_{step}" / "huggingface")
        record = update_run(record, "validated", eligible=True, validation={"log": str(log)}, inference_model=inference_model)
        print(f"Training is ready: {run_id}\nNext: ./root-sft infer --run {run_id}")
        return 0
    except (RuntimeError, ValueError) as error:
        log.write_text(str(error) + "\n", encoding="utf-8")
        fail_run(record, "training", 2, log, str(error))
        print(f"Cannot train: {error}", file=sys.stderr)
        return 2


def cmd_infer(args: argparse.Namespace) -> int:
    if args.model:
        if not args.run:
            raise ValueError("Choose a name for a base-model run with --run NAME.")
        record = create_run(args.run, model=args.model, seed=args.seed, command=sys.argv[1:], kind="base")
        record = update_run(record, "validated", eligible=True, validation={"kind": "base model"})
        model_path = args.model
    else:
        if not args.run:
            raise ValueError("Choose --run NAME or --run latest.")
        record = load_run(args.run, require_eligible=True)
        model_path = args.model_path or str(saved_path(record.get("inference_model") or record["paths"]["checkpoint"]))
    log = command_log(record, "infer")
    try:
        require_verl_container()
        record = update_run(record, "inferred")
        env = os.environ.copy()
        env.update({"MODEL": model_path, "PROMPTS": str(saved_path(record["paths"]["prompts"]) / "test.jsonl"), "COMPLETIONS": str(saved_path(record["paths"]["completions"]) / "test.jsonl")})
        code = execute(["bash", str(REPO_ROOT / "inference/run_atlas_sft_inference.sh")], log=log, env=env, dry_run=args.dry_run)
        if code:
            fail_run(record, "inference", code, log, "Inference stopped before completions were written.")
            return code
        update_run(record, "inferred", eligible=True, completions=stored_path(Path(env["COMPLETIONS"])))
        print(f"Completions are ready: {env['COMPLETIONS']}\nNext: ./root-sft score --run {record['id']}")
        return 0
    except RuntimeError as error:
        log.write_text(str(error) + "\n", encoding="utf-8")
        fail_run(record, "inference", 2, log, str(error))
        print(f"Cannot infer: {error}", file=sys.stderr)
        return 2


def cmd_finalize(args: argparse.Namespace) -> int:
    """Export a checkpoint after a trainer completed but its shell handoff failed."""
    record = load_run(args.run)
    checkpoint_root = saved_path(record["paths"]["checkpoint"])
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or not tracker.read_text(encoding="utf-8").strip().isdigit():
        raise ValueError("No completed checkpoint was found for this run.")
    step = tracker.read_text(encoding="utf-8").strip()
    checkpoint = checkpoint_root / f"global_step_{step}"
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory is missing: {checkpoint}")
    log = command_log(record, "finalize")
    require_verl_container()
    record = update_run(record, "checkpointed")
    env = os.environ.copy()
    base_model = record["model_revision"]["resolved"]
    export = ["uv", "run", "--project", str(REPO_ROOT / "verl"), "--frozen", "--extra", "fsdp", "--extra", "sglang", "python", str(REPO_ROOT / "inference/export_verl_lora_adapter.py"), "--checkpoint", str(checkpoint), "--base-model", base_model]
    code = execute(export, log=log, env=env, dry_run=args.dry_run)
    if code:
        fail_run(record, "adapter export", code, log, "Checkpoint export failed.")
        return code
    record = update_run(record, "exported")
    validate = ["uv", "run", "--project", str(REPO_ROOT / "verl"), "--frozen", "--extra", "fsdp", "--extra", "sglang", "python", str(REPO_ROOT / "inference/validate_model_export.py"), str(checkpoint / "huggingface")]
    code = execute(validate, log=log, env=env, dry_run=args.dry_run)
    if code:
        fail_run(record, "export validation", code, log, "The exported checkpoint did not validate.")
        return code
    update_run(record, "validated", eligible=True, inference_model=stored_path(checkpoint / "huggingface"), validation={"log": stored_path(log), "checkpoint_step": int(step)})
    print(f"Checkpoint export is ready. Next: ./root-sft infer --run {record['id']}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    record = load_run(args.run, require_eligible=True)
    completions = saved_path(record.get("completions") or saved_path(record["paths"]["completions"]) / "test.jsonl")
    if not args.dry_run and not completions.is_file():
        raise ValueError(f"No completions found for '{record['id']}'. Run './root-sft infer --run {record['id']}' first.")
    if not shutil.which("root") and not args.use_cvmfs_root:
        raise RuntimeError("Scoring needs ROOT. Start the ATLAS ROOT shell, run lsetup, then retry; use --use-cvmfs-root only for the fallback runtime.")
    apptainer_isolated = bool(os.environ.get("APPTAINER_CONTAINER"))
    if not shutil.which("bwrap") and not (getattr(args, "allow_unsandboxed_score", False) or apptainer_isolated) and not args.dry_run:
        raise RuntimeError("Scoring generated commands needs bubblewrap isolation. Install bwrap, or explicitly use --allow-unsandboxed-score in an already isolated environment.")
    report_interpreter = report_python(getattr(args, "report_python", None), dry_run=args.dry_run)
    log = command_log(record, "score")
    output = saved_path(record["paths"]["benchmark"])
    record = update_run(record, "scored")
    env = os.environ.copy()
    if not shutil.which("root"):
        env["ROOT_USE_CURRENT"] = "0"
    else:
        env["ROOT_USE_CURRENT"] = "1"
    env["PYTHON_BIN"] = report_interpreter
    with tempfile.TemporaryDirectory(prefix=f"root-sft-score-{record['id']}-") as sandbox:
        env["TMPDIR"] = sandbox
        # The benchmark evaluates model-produced commands. Give it a read-only
        # view of the host and writable mounts only for its temporary directory
        # and this run's benchmark output.
        score_command = ["bash", str(REPO_ROOT / "inference/score_atlas_benchmark.sh"), str(completions), record["id"], str(output)]
        if shutil.which("bwrap"):
            score_command = [
                "bwrap", "--die-with-parent", "--new-session", "--unshare-net",
                "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                "--bind", str(output), str(output), "--bind", sandbox, sandbox,
                "--setenv", "TMPDIR", sandbox, "--setenv", "HOME", sandbox,
                "--chdir", sandbox,
            ] + score_command
        code = execute(score_command, log=log, cwd=Path(sandbox), timeout=args.timeout, env=env, dry_run=args.dry_run)
    if code:
        fail_run(record, "scoring", code, log, "Scoring stopped before a report was written.")
        return code
    update_run(record, "scored", eligible=True, report=stored_path(output / f"{record['id']}-report.json"), summary=stored_path(output / f"{record['id']}-summary.csv"), evaluation={"timeout_seconds": args.timeout, "log": stored_path(log)})
    print(f"Score report is ready: {output}\nNext: ./root-sft plot --run {record['id']}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    record = load_run(args.run, require_eligible=True)
    summary = saved_path(record.get("summary") or saved_path(record["paths"]["benchmark"]) / f"{record['id']}-summary.csv")
    if not args.dry_run and not summary.is_file():
        raise ValueError(f"No score summary found for '{record['id']}'. Run './root-sft score --run {record['id']}' first.")
    log = command_log(record, "plot")
    plot_dir = saved_path(record["paths"]["plots"])
    command = [sys.executable, str(DATASET_ROOT / "tools/benchmark/plot_report.py"), "--summary", str(summary), "--output-dir", str(plot_dir), "--model", f"{record['id']}:{record['model']}:#4C78A8"]
    code = execute(command, log=log, dry_run=args.dry_run)
    if code:
        fail_run(record, "plotting", code, log, "Plotting stopped before images were written.")
        return code
    record = update_run(record, "plotted", eligible=True, plots=stored_path(plot_dir))
    update_run(record, "ready", eligible=True)
    print(f"Plots are ready: {plot_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    record = load_run(args.run)
    benchmark = saved_path(record["paths"]["benchmark"])
    executed = benchmark / f"{record['id']}-executed.jsonl"
    if not executed.is_file():
        raise ValueError("No executed completions were found. Run score first.")
    log = command_log(record, "report")
    command = ["uv", "run", "--project", str(REPO_ROOT), "--extra", "reports", "--locked", "python", str(DATASET_ROOT / "tools/benchmark/build_report.py"), "--dataset", str(DATASET_ROOT / "data/sft/test.parquet"), "--expected", str(DATASET_ROOT / "benchmark/expected-results/test.jsonl"), "--completion", f"{record['id']}={executed}", "--output-json", str(benchmark / f"{record['id']}-report.json"), "--output-csv", str(benchmark / f"{record['id']}-report.csv"), "--summary-csv", str(benchmark / f"{record['id']}-summary.csv")]
    code = execute(command, log=log, dry_run=args.dry_run)
    if code:
        return code
    update_run(record, "scored", eligible=True, report=stored_path(benchmark / f"{record['id']}-report.json"), summary=stored_path(benchmark / f"{record['id']}-summary.csv"))
    print(f"Report is ready. Next: ./root-sft plot --run {record['id']}")
    return 0


def cmd_runs(_: argparse.Namespace) -> int:
    registry = read_registry()
    if not registry["runs"]:
        print("No runs yet. Start one with './root-sft train --model qwen3.5-0.8b --epochs 1'.")
        return 0
    print(f"{'RUN':32} {'STEP':12} {'MODEL':18} UPDATED")
    for run_id, item in sorted(registry["runs"].items(), key=lambda pair: pair[1]["updated_at"], reverse=True):
        print(f"{run_id:32} {item['phase']:12} {item['model']:18} {item['updated_at']}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    print(json.dumps(load_run(args.run), indent=2, sort_keys=True))
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    record = load_run(args.run)
    directory = saved_path(record["paths"]["logs"])
    path = directory / f"{args.phase}.log" if args.phase else max(directory.glob("*.log"), key=lambda item: item.stat().st_mtime, default=None)
    if path is None or not path.is_file():
        raise ValueError("No log was found for that run and phase.")
    print(path.read_text(encoding="utf-8"), end="")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    record = load_run(args.run)
    if record["phase"] != "failed":
        raise ValueError(f"Run '{record['id']}' has not failed and does not need resume.")
    settings = record.get("training")
    if not settings:
        raise ValueError("This failed run was not a training run. Inspect its log with './root-sft logs RUN'.")
    return cmd_train(argparse.Namespace(model=record["model"], epochs=settings["epochs"], learning_rate=settings.get("learning_rate", 1e-5), batch_size=settings.get("batch_size", 16), lora_rank=settings.get("lora_rank", 16), lora_alpha=settings.get("lora_alpha", 16), run=record["id"], seed=record.get("seed"), resume=True, force=False, dry_run=args.dry_run))


def cmd_compare(args: argparse.Namespace) -> int:
    first, second = (load_run(value) for value in (args.first, args.second))
    print(f"Comparing {first['id']} with {second['id']}")
    for key in ("model", "seed", "phase"):
        print(f"{key}: {first.get(key)} | {second.get(key)}")
    print(f"dataset revision: {first['dataset'].get('revision')} | {second['dataset'].get('revision')}")
    for record in (first, second):
        report_path = record.get("report")
        if not report_path or not Path(report_path).is_file():
            print(f"{record['id']} score: not scored")
            continue
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        model_score = report.get("models", {}).get(record["id"], {})
        if model_score:
            print(f"{record['id']} score: {model_score.get('passed')}/{report.get('total')} passed ({model_score.get('accuracy', 0):.1%})")
        else:
            print(f"{record['id']} score: report is present but has no matching run label")
    return 0


def parser() -> argparse.ArgumentParser:
    formatter = argparse.RawDescriptionHelpFormatter
    root = argparse.ArgumentParser(
        prog="root-sft", description="Train, test, score, and plot ROOT SFT runs.",
        epilog="Example:\n  ./root-sft check\n  ./root-sft train --model qwen3.5-0.8b --epochs 10 --run my-run",
        formatter_class=formatter,
    )
    root.add_argument("--verbose", action="store_true", help="Show implementation details when a command supports them.")
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Check what this machine can do.", description="Check the repository, data, workspace, and optional runtimes.", epilog="Run on: host.\nExample: ./root-sft check", formatter_class=formatter)
    check.set_defaults(func=cmd_check)
    train = commands.add_parser("train", help="Train a ROOT SFT model in the GPU container.", description="Train a named model run and save its checkpoint and details together. Console output is also saved to the run log.", epilog="Run on: GPU container.\nExample: ./root-sft train --model qwen3.5-0.8b --epochs 3 --learning-rate 1e-4 --batch-size 8 --run my-run\nOutput: a validated run when export succeeds.", formatter_class=formatter)
    train.add_argument("--model", required=True, choices=MODEL_PROFILES)
    train.add_argument("--epochs", required=True, type=int)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lora-rank", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--run"); train.add_argument("--seed", type=int); train.add_argument("--resume", action="store_true"); train.add_argument("--force", action="store_true"); train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=cmd_train)
    infer = commands.add_parser("infer", help="Create held-out completions in the GPU container.", description="Generate held-out answers for a saved run or a named base-model run.", epilog="Run on: GPU container.\nExample: ./root-sft infer --run my-run\nOutput: completions/test.jsonl in the run directory.", formatter_class=formatter)
    infer.add_argument("--run"); infer.add_argument("--model"); infer.add_argument("--model-path"); infer.add_argument("--seed", type=int); infer.add_argument("--dry-run", action="store_true")
    infer.set_defaults(func=cmd_infer)
    finalize = commands.add_parser("finalize", help="Export a completed checkpoint without retraining.", description="Recover a saved checkpoint when training completed but export did not run.", epilog="Run on: GPU container.\nExample: ./root-sft finalize --run my-run", formatter_class=formatter)
    finalize.add_argument("--run", required=True); finalize.add_argument("--dry-run", action="store_true"); finalize.set_defaults(func=cmd_finalize)
    score = commands.add_parser("score", help="Score a run in an ATLAS ROOT shell.", description="Run the benchmark scorer for saved completions in an isolated workspace.", epilog="Run on: ATLAS ROOT shell.\nExample: ./root-sft score --run my-run\nOutput: benchmark report and summary files in the run directory.", formatter_class=formatter)
    score.add_argument("--run", required=True); score.add_argument("--timeout", type=int, default=300); score.add_argument("--use-cvmfs-root", action="store_true"); score.add_argument("--report-python"); score.add_argument("--dry-run", action="store_true")
    score.add_argument("--allow-unsandboxed-score", action="store_true", help="Allow scoring without bwrap when the current container is already isolated.")
    score.set_defaults(func=cmd_score)
    report = commands.add_parser("report", help="Build a report from completed ROOT evaluation.", description="Build JSON and CSV reports without rerunning ROOT evaluation.", epilog="Run on: host.\nExample: ./root-sft report --run my-run", formatter_class=formatter)
    report.add_argument("--run", required=True); report.add_argument("--dry-run", action="store_true"); report.set_defaults(func=cmd_report)
    plot = commands.add_parser("plot", help="Plot a score report with host-side Matplotlib.", description="Render score-report images without ROOT or a container.", epilog="Run on: host.\nExample: ./root-sft plot --run my-run\nOutput: PNG and PDF images in the run's plots directory.", formatter_class=formatter)
    plot.add_argument("--run", required=True); plot.add_argument("--dry-run", action="store_true"); plot.set_defaults(func=cmd_plot)
    runs = commands.add_parser("runs", help="List saved runs.", description="List named runs and their current step.", epilog="Run on: host.\nExample: ./root-sft runs", formatter_class=formatter); runs.set_defaults(func=cmd_runs)
    show = commands.add_parser("show", help="Show one run's saved details.", description="Print the saved details for one run.", epilog="Run on: host.\nExample: ./root-sft show my-run", formatter_class=formatter); show.add_argument("run"); show.set_defaults(func=cmd_show)
    logs = commands.add_parser("logs", help="Show output from one run.", description="Print the newest log or one chosen step from a run.", epilog="Run on: host.\nExample: ./root-sft logs my-run --phase train", formatter_class=formatter); logs.add_argument("run"); logs.add_argument("--phase"); logs.set_defaults(func=cmd_logs)
    resume = commands.add_parser("resume", help="Continue a failed training run.", description="Continue a failed training run with its saved settings.", epilog="Run on: GPU container.\nExample: ./root-sft resume my-run", formatter_class=formatter); resume.add_argument("run"); resume.add_argument("--dry-run", action="store_true"); resume.set_defaults(func=cmd_resume)
    compare = commands.add_parser("compare", help="Compare two saved runs.", description="Compare model, data, settings, and available scores.", epilog="Run on: host.\nExample: ./root-sft compare base-run trained-run", formatter_class=formatter); compare.add_argument("first"); compare.add_argument("second"); compare.set_defaults(func=cmd_compare)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except (ValueError, FileExistsError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Cannot continue: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
