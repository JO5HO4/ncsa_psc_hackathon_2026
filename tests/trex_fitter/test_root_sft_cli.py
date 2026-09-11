"""CPU-only checks for the ROOT SFT command interface."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import root_sft  # noqa: E402


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    for split in ("train", "validation", "test"):
        path = dataset / "data" / "sft" / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{split} fixture".encode())
    artifacts = tmp_path / "artifacts" / "atlas-open-data-sft"
    monkeypatch.setattr(root_sft, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(root_sft, "DATASET_ROOT", dataset)
    monkeypatch.setattr(root_sft, "ARTIFACT_ROOT", artifacts)
    monkeypatch.setattr(root_sft, "RUNS_ROOT", artifacts / "runs")
    monkeypatch.setattr(root_sft, "REGISTRY_PATH", artifacts / "runs.json")
    monkeypatch.setattr(root_sft, "LOCK_PATH", artifacts / ".runs.lock")
    monkeypatch.setenv("ROOT_SFT_IN_CONTAINER", "1")
    return tmp_path


def test_run_registry_resolves_only_validated_runs(workflow):
    first = root_sft.create_run("first", model="qwen3.5-0.8b", seed=4, command=["train"], kind="trained")
    second = root_sft.create_run("second", model="qwen3.5-0.8b", seed=5, command=["train"], kind="trained")
    root_sft.update_run(first, "validated", eligible=True)

    assert root_sft.load_run("latest")["id"] == "first"
    assert root_sft.load_run("second")["eligible"] is False
    assert json.loads((root_sft.run_path("first") / "run.json").read_text())["dataset"]["inputs"]["train"]["sha256"]
    assert not Path(first["paths"]["run"]).is_absolute()
    assert not Path(json.loads(root_sft.REGISTRY_PATH.read_text())["runs"]["first"]["path"]).is_absolute()


def test_train_dry_run_records_train_and_validation_only(workflow):
    args = Namespace(model="qwen3.5-0.8b", epochs=1, run="smoke", seed=9, resume=False, force=False, dry_run=True)

    assert root_sft.cmd_train(args) == 0
    record = root_sft.load_run("smoke", require_eligible=True)
    assert record["phase"] == "validated"
    assert record["dataset"]["inputs"]["test"]["path"].endswith("test.parquet")
    assert record["dataset"]["inputs"]["train"]["path"].endswith("train.parquet")
    assert (root_sft.saved_path(record["paths"]["logs"]) / "train.log").is_file()


def test_train_recovers_a_completed_checkpoint_after_export_handoff_fails(workflow, monkeypatch):
    def fake_execute(command, **kwargs):
        return 1 if command[-1].endswith("run_verl_sft.sh") else 0

    monkeypatch.setattr(root_sft, "execute", fake_execute)
    record = root_sft.create_run("recovered", model="qwen3.5-0.8b", seed=None, command=["train"], kind="trained")
    root_sft.fail_run(record, "training", 1, root_sft.run_path("recovered") / "logs" / "train.log", "fixture failure")
    checkpoint = root_sft.run_path("recovered") / "checkpoint"
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("12\n")
    (checkpoint / "global_step_12").mkdir()

    args = Namespace(model="qwen3.5-0.8b", epochs=1, run="recovered", seed=None, resume=True, force=False, dry_run=False)

    assert root_sft.cmd_train(args) == 0
    record = root_sft.load_run("recovered", require_eligible=True)
    assert record["phase"] == "validated"
    assert record["validation"]["checkpoint_step"] == 12


def test_named_run_cannot_be_replaced_without_resume(workflow):
    root_sft.create_run("kept", model="qwen3.5-0.8b", seed=None, command=["train"], kind="trained")

    with pytest.raises(FileExistsError, match="resume"):
        root_sft.create_run("kept", model="qwen3.5-0.8b", seed=None, command=["train"], kind="trained")


def test_resume_reuses_only_a_failed_training_run(workflow):
    record = root_sft.create_run("retry", model="qwen3.5-0.8b", seed=7, command=["train"], kind="trained")
    record = root_sft.update_run(record, "training", training={"epochs": 1, "profile": "0.8b"})
    root_sft.fail_run(record, "training", 1, Path(record["paths"]["logs"]) / "train.log", "fixture failure")

    assert root_sft.cmd_resume(Namespace(run="retry", dry_run=True)) == 0
    assert root_sft.load_run("retry", require_eligible=True)["phase"] == "validated"


def test_base_model_inference_creates_a_tracked_run(workflow):
    args = Namespace(run="base", model="Qwen/Qwen3.5-0.8B", model_path=None, seed=3, dry_run=True)

    assert root_sft.cmd_infer(args) == 0
    record = root_sft.load_run("base", require_eligible=True)
    assert record["kind"] == "base"
    assert record["phase"] == "inferred"


def test_score_and_plot_dry_run_keep_outputs_in_the_run(workflow, monkeypatch):
    record = root_sft.create_run("scored", model="qwen3.5-0.8b", seed=None, command=["infer"], kind="trained")
    root_sft.update_run(record, "inferred", eligible=True, completions=str(root_sft.run_path("scored") / "completions" / "test.jsonl"))
    monkeypatch.setattr(root_sft.shutil, "which", lambda name: "/usr/bin/root" if name == "root" else None)

    assert root_sft.cmd_score(Namespace(run="scored", timeout=1, use_cvmfs_root=False, dry_run=True)) == 0
    assert root_sft.cmd_plot(Namespace(run="scored", dry_run=True)) == 0
    record = root_sft.load_run("scored", require_eligible=True)
    assert record["phase"] == "ready"
    assert record["summary"].startswith(record["paths"]["benchmark"])
    assert record["plots"] == record["paths"]["plots"]
