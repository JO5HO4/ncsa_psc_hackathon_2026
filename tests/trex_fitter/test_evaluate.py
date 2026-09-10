"""Execution-reporting tests; no real native fit or dataset generation."""

import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trex_fitter import evaluate

ROOT = Path(__file__).resolve().parents[2]
HYY = ROOT / "data/configs/examples/hyy.config"


def test_mock_execution_report():
    result = subprocess.run(
        [sys.executable, "-m", "trex_fitter.evaluate", str(HYY), "--mock"],
        cwd=ROOT, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["success"] and report["mock"]
    assert report["verification"]["analysis_valid"]
    assert report["significance"] == 1.7


@pytest.mark.parametrize("mode,exit_code,success,significance", [
    ("success", 0, True, 1.9),
    ("no_significance", 0, True, None),
    ("failure", 2, False, None),
    ("timeout", -15, False, None),
    ("kill", -9, False, None),
])
def test_native_report_and_process_group_cleanup(tmp_path, monkeypatch, capsys,
                                                mode, exit_code, success, significance):
    monkeypatch.setattr(sys, "argv", ["evaluate", str(HYY), "--log-dir", str(tmp_path), "--timeout", "1"])
    monkeypatch.setattr(evaluate.runner.podman_trex, "to_container_path", str)
    # A stale significance must not contaminate this invocation's result.
    (tmp_path / "evaluation.stdout.log").write_text("SIGNIFICANCE = 99\n")
    waits = [exit_code]
    if mode in {"timeout", "kill"}:
        waits.insert(0, subprocess.TimeoutExpired("test", 1))
    if mode == "kill":
        waits.insert(1, subprocess.TimeoutExpired("test", 5))
    process = SimpleNamespace(pid=123456, returncode=exit_code, wait=Mock(side_effect=waits))
    def popen(command, **kwargs):
        assert kwargs["start_new_session"] is True
        assert kwargs["cwd"] == ROOT
        assert "runner.py" in command[1]
        kwargs["stdout"].write("Observed significance (median): 1.9\n" if mode == "success" else "new run\n")
        return process
    monkeypatch.setattr(evaluate.subprocess, "Popen", popen)
    killpg = Mock()
    monkeypatch.setattr(evaluate.os, "killpg", killpg)
    with pytest.raises(SystemExit) as stop:
        evaluate.main()
    assert stop.value.code == (0 if success else 1)
    report = json.loads(capsys.readouterr().out)
    assert report["success"] is success
    assert report["returncode"] == exit_code
    assert report["significance"] == significance
    assert report["timed_out"] == (mode in {"timeout", "kill"})
    assert report["elapsed_seconds"] >= 0
    expected_signals = [signal.SIGTERM, signal.SIGKILL] if mode == "kill" else [signal.SIGTERM] if mode == "timeout" else []
    assert [call.args for call in killpg.call_args_list] == [(123456, sig) for sig in expected_signals]


def test_invalid_analysis_does_not_execute(tmp_path, monkeypatch, capsys):
    config = tmp_path / "bad.config"
    config.write_text(HYY.read_text().replace('POI: "mu_H"', 'POI: "undeclared"'))
    monkeypatch.setattr(sys, "argv", ["evaluate", str(config), "--log-dir", str(tmp_path)])
    monkeypatch.setattr(evaluate.runner, "config_path", lambda value: Path(value))
    monkeypatch.setattr(evaluate.runner.podman_trex, "to_container_path", str)
    execute = Mock(side_effect=AssertionError("invalid config must not execute"))
    monkeypatch.setattr(evaluate.subprocess, "Popen", execute)
    with pytest.raises(SystemExit) as stop:
        evaluate.main()
    assert stop.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["verification"]["analysis_valid"]
    assert not report["success"] and report["returncode"] is None
    execute.assert_not_called()
