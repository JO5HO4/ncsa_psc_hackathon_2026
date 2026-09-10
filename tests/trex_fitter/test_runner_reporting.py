"""One CLI: real subprocess reporting/timeout tests with test-only fixtures."""

import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from trex_fitter import runner, runtime

ROOT = Path(__file__).resolve().parents[2]
HYY = ROOT / "data/configs/examples/hyy.config"
FAKE = Path(__file__).parent / "fixtures/fake_process.py"


@pytest.mark.parametrize("mode,expected,code", [("success", 1.9, 0), ("failure", None, 1), ("empty", None, 0)])
def test_json_stdout_current_run_only(tmp_path, monkeypatch, capsys, mode, expected, code):
    (tmp_path / "old.log").write_text("SIGNIFICANCE=99")
    monkeypatch.setattr(sys, "argv", ["runner", str(HYY), "--json", "-", "--log-dir", str(tmp_path)])
    monkeypatch.setattr(runner, "worker_command", lambda args, logs: [sys.executable, str(FAKE), mode])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == code
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["significance"] == expected
    assert report["success"] == (code == 0)
    assert report["returncode"] == (7 if mode == "failure" else 0)
    assert "fixture stdout" in captured.err and "fixture stderr" in captured.err
    assert Path(report["log_dir"]).parent == tmp_path
    assert "fixture stdout" in (Path(report["log_dir"]) / "runner.stdout.log").read_text()


@pytest.mark.parametrize("mode", ["descendant", "orphan"])
def test_full_chain_timeout_kills_descendants_and_cleans_container(tmp_path, monkeypatch, capsys, mode):
    pid_file = tmp_path / "pid"
    report_path = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["runner", str(HYY), "--backend", "coffea", "--actions", "nwsf",
                                      "--timeout", "1", "--json", str(report_path), "--log-dir", str(tmp_path)])
    cid = "a" * 64
    def command(args, log_dir):
        (log_dir / "container-test.cid").write_text(cid)
        return [sys.executable, str(FAKE), mode, str(pid_file)]
    monkeypatch.setattr(runner, "worker_command", command)
    remove = Mock()
    monkeypatch.setattr(runtime, "remove_container", remove)
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 124
    report = json.loads(report_path.read_text())
    assert report["timed_out"] and not report["success"]
    assert report["significance"] is None
    remove.assert_called_once()
    assert remove.call_args.args[0].read_text() == cid
    child = Path("/proc") / pid_file.read_text() / "stat"
    # An orphan can briefly remain a zombie until the system reaps it.
    def still_running():
        try:
            return child.read_text().split()[2] != "Z"
        except (FileNotFoundError, ProcessLookupError):
            return False
    deadline = time.monotonic() + 2
    while still_running() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not still_running()


def test_report_launch_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["runner", str(HYY), "--json", "-", "--log-dir", str(tmp_path)])
    monkeypatch.setattr(runner, "worker_command", lambda args, logs: [str(tmp_path / "absent")])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["returncode"] is None and report["error"]


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_timeout_must_be_positive_finite(timeout, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runner", str(HYY), "--timeout", timeout])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 2


def test_actual_cli_json_dry_run(tmp_path):
    result = subprocess.run([sys.executable, "-m", "trex_fitter.runner", str(HYY),
                             "--backend", "coffea", "--actions", "nwsf", "--dry-run",
                             "--json", "-", "--timeout", "10", "--log-dir", str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["success"] and report["dry_run"]
    assert report["actions"] == list("nwsf")
    assert "native container action string: wsf" in result.stderr
    assert report["significance"] is None


def test_native_output_directory_is_honored(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runner", str(HYY), "--actions", "wsf", "--output-dir", str(tmp_path)])
    execute = Mock()
    monkeypatch.setattr(runner, "run_actions", execute)
    runner.main()
    assert execute.call_args.kwargs["work_dir"] == tmp_path


def test_container_cleanup_validates_identity(tmp_path, monkeypatch):
    cid = tmp_path / "container.cid"
    cid.write_text("not-an-id")
    with pytest.raises(ValueError):
        runtime.remove_container(cid)
    cid.write_text("a" * 64)
    execute = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(runtime.subprocess, "run", execute)
    runtime.remove_container(cid)
    assert execute.call_args.args[0] == [runtime.CONTAINER_ENGINE, "rm", "--force", "--ignore", "a" * 64]


def test_nonfinite_significance_is_not_a_json_number():
    assert runtime.parse_significance_from_text("SIGNIFICANCE=1e999") is None


def test_no_public_mock_option(tmp_path):
    result = subprocess.run([sys.executable, "-m", "trex_fitter.runner", str(HYY), "--mock"],
                            cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
