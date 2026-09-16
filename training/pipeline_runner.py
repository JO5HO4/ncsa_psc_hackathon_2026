"""Config-driven pipeline runner: baseline inference -> LoRA SFT -> post-training inference.

Runs INSIDE the verl container -- see training/run_pipeline.sh, which reads
training/pipeline.conf, does the GPU/container launch, and invokes this
script as `python training/pipeline_runner.py <run_dir>`. Each of the three
stages is independently toggleable via env vars sourced from the config file
(see training/pipeline.conf.example). Never executes generated ROOT commands
or runs ROOT scoring itself -- that is a separate, host-side, ROOT-enabled
step performed by run_pipeline.sh after this script exits (see
inference/score_atlas_benchmark.sh), because scoring needs CVMFS/ROOT, not
the GPU/verl container.

Unlike training/root_before_after.py (which always runs all three stages
against the fixed ATLAS-command dataset and produced the results cited in
results/*.md), this script is stage-toggleable and works against any
train/validation/test files -- including a fresh release from
training/translator/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path("/workspace")
sys.path.insert(0, str(REPO_ROOT))

from training.translator.task_profiles import get_task_profile  # noqa: E402

# Task profiles with an execution contract (an expected_result to diff a
# literally-executed command's output against). See inference/
# score_atlas_benchmark.sh, which cannot be made generic -- only these
# profiles' data has the shape it requires.
EXECUTABLE_TASK_PROFILES = {"root_command"}

# Mirrors training/scripts/qwen35_profile.sh's size->model mapping. Kept here
# (rather than sourcing the bash script from Python) because it is small and
# only used to resolve the exact snapshot used consistently across stages;
# QWEN35_MODEL_SIZE is still passed through to run_verl_sft.sh unchanged, so
# its own profile resolution (GPU defaults, etc.) is unaffected.
QWEN35_MODEL_PATHS = {"0.8b": "Qwen/Qwen3.5-0.8B", "9b": "Qwen/Qwen3.5-9B"}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def run_stage(name: str, command: list, env: dict, output: Path) -> None:
    status = {"stage": name, "started": datetime.now(timezone.utc).isoformat(), "command": command}
    path = output / f"{name}-status.json"
    path.write_text(json.dumps(status, indent=2))
    print(f"START {name}; log: {output / (name + '.log')}", flush=True)
    with (output / f"{name}.log").open("x") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    status.update(exit_code=result.returncode, finished=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(status, indent=2))
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit {result.returncode}; inspect its log")
    print(f"PASS {name}", flush=True)


def extract_prompts(eval_file: Path, output: Path) -> list:
    """Turn a translator-shaped {id, messages:[system,user,assistant]} JSONL
    into {id, prompt} rows for inference/run_prompts.py. Written once and
    reused verbatim for both baseline and post-training inference, so both
    stages answer exactly the same questions."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    prompts = []
    for line in eval_file.read_text().splitlines():
        if not line:
            continue
        record = json.loads(line)
        users = [message["content"] for message in record["messages"] if message["role"] == "user"]
        if len(users) != 1:
            raise ValueError(f"{record.get('id', '<unknown>')}: expected exactly one user message")
        prompts.append({"id": record["id"], "prompt": users[0]})
    if len(prompts) != len({p["id"] for p in prompts}):
        raise ValueError(f"Duplicate ids in {eval_file}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(prompt) + "\n" for prompt in prompts))
    return prompts


def validate_config(config: dict) -> None:
    if not any(config[key] for key in ("run_baseline_inference", "run_training", "run_post_training_inference")):
        raise ValueError(
            "At least one of RUN_BASELINE_INFERENCE, RUN_TRAINING, "
            "RUN_POST_TRAINING_INFERENCE must be true"
        )
    if config["run_post_training_inference"] and not config["run_training"]:
        existing = config.get("existing_model")
        if not existing or not Path(existing).exists():
            raise ValueError(
                "RUN_POST_TRAINING_INFERENCE=true with RUN_TRAINING=false requires "
                "EXISTING_MODEL to point at an existing exported model"
            )
    if config["run_root_scoring"] and config["task_profile"] not in EXECUTABLE_TASK_PROFILES:
        raise ValueError(
            f"RUN_ROOT_SCORING=true requires an executable task profile "
            f"({sorted(EXECUTABLE_TASK_PROFILES)}); got {config['task_profile']!r}"
        )


def load_config_from_env() -> tuple:
    task_profile_name = os.environ["TASK_PROFILE"]
    task_profile = get_task_profile(task_profile_name)  # raises ValueError early if unknown
    config = dict(
        run_baseline_inference=env_flag("RUN_BASELINE_INFERENCE"),
        run_training=env_flag("RUN_TRAINING"),
        run_post_training_inference=env_flag("RUN_POST_TRAINING_INFERENCE"),
        run_root_scoring=env_flag("RUN_ROOT_SCORING"),
        task_profile=task_profile_name,
        existing_model=os.environ.get("EXISTING_MODEL") or None,
    )
    validate_config(config)
    return config, task_profile


def resolve_base_model(environment: dict) -> str:
    model_path = environment.get("MODEL_PATH")
    if model_path:
        return model_path
    size = environment.get("QWEN35_MODEL_SIZE")
    try:
        return QWEN35_MODEL_PATHS[size]
    except KeyError:
        raise ValueError(
            f"Set MODEL_PATH directly, or QWEN35_MODEL_SIZE to one of {sorted(QWEN35_MODEL_PATHS)}; got {size!r}"
        ) from None


def resolve_base_snapshot(base_model: str, revision: Optional[str]) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=base_model, revision=revision)


def run_inference(
    *, model_path, base_model_for_adapter, prompts_path, system_prompt, output_path, name, env, run_dir,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(REPO_ROOT / "inference/run_prompts.py"),
        "--model", str(model_path), "--prompts", str(prompts_path),
        "--prompt-field", "prompt", "--id-field", "id", "--format", "chat",
        "--no-enable-thinking", "--system-prompt", system_prompt,
        "--device", "cuda", "--temperature", "0", "--max-new-tokens", "256",
        "--output", str(output_path),
    ]
    if base_model_for_adapter:
        command += ["--base-model", base_model_for_adapter]
    run_stage(name, command, env, run_dir)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: pipeline_runner.py RUN_DIR")
    run_dir = Path(sys.argv[1]).resolve()
    config, task_profile = load_config_from_env()
    environment = os.environ.copy()

    base_model = resolve_base_model(environment)
    base_revision = environment.get("BASE_REVISION") or None
    base_snapshot = resolve_base_snapshot(base_model, base_revision)

    prompts_path = run_dir / "prompts.jsonl"
    needs_prompts = config["run_baseline_inference"] or config["run_post_training_inference"]
    prompts = None
    if needs_prompts:
        eval_file = Path(environment["EVAL_FILE"])
        prompts = extract_prompts(eval_file, prompts_path)
        print(f"Extracted {len(prompts)} prompts from {eval_file}", flush=True)

    if config["run_baseline_inference"]:
        run_inference(
            model_path=base_snapshot, base_model_for_adapter=None,
            prompts_path=prompts_path, system_prompt=task_profile.system_prompt,
            output_path=run_dir / "baseline" / "completions.jsonl",
            name="baseline-inference", env=environment, run_dir=run_dir,
        )

    export_dir = None
    if config["run_training"]:
        save_dir = run_dir / "checkpoints"
        training_env = environment.copy()
        training_env.update(MODEL_PATH=base_snapshot, SAVE_DIR=str(save_dir))
        run_stage("train", ["bash", str(REPO_ROOT / "training/scripts/run_verl_sft.sh")], training_env, run_dir)
        tracker = save_dir / "latest_checkpointed_iteration.txt"
        step = tracker.read_text().strip()
        if not step.isdigit():
            raise ValueError(f"Invalid checkpoint step in {tracker}")
        export_dir = save_dir / f"global_step_{step}" / "huggingface"
        if not export_dir.is_dir():
            raise FileNotFoundError(f"Expected export directory missing: {export_dir}")

    if config["run_post_training_inference"]:
        model_for_inference = export_dir if export_dir is not None else Path(config["existing_model"])
        run_inference(
            model_path=model_for_inference, base_model_for_adapter=base_snapshot,
            prompts_path=prompts_path, system_prompt=task_profile.system_prompt,
            output_path=run_dir / "trained" / "completions.jsonl",
            name="trained-inference", env=environment, run_dir=run_dir,
        )

    summary = {
        "run_name": environment.get("RUN_NAME"),
        "task_profile": config["task_profile"],
        "stages": {key: value for key, value in config.items() if key.startswith("run_")},
        "prompt_count": len(prompts) if prompts is not None else None,
        "base_model": base_model,
        "base_snapshot": base_snapshot,
        "export_dir": str(export_dir) if export_dir else None,
        "existing_model": config["existing_model"],
        "root_execution_scoring": "not_run",
    }
    (run_dir / "pipeline-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Pipeline complete: {run_dir}")


if __name__ == "__main__":
    main()
