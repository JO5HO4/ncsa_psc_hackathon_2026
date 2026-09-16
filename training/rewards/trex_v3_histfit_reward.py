"""VERL scalar reward for the terminal v3-histfit-001 config task.

Model output is config text only. Every call gets a fresh workspace and the
task evaluator owns the native execution; model text is never shell or Python.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from trex_fitter.rl_tasks import v3_histfit_001  # noqa: E402


MAX_CONFIG_CHARS = 24_000
DEFAULT_TIMEOUT_SECONDS = 1800
FORMAT_REWARD_CAP = 0.05


def _format_checks(response: str) -> dict[str, bool]:
    """Cheap pre-parse curriculum signals for an otherwise sparse task.

    These checks deliberately carry little weight.  They teach an untrained
    policy to emit TREx-like config syntax, while a config that parses and
    passes the semantic checks still earns substantially more reward.
    """
    text = response.strip()
    return {
        "job_header": bool(re.search(r"(?mi)^\s*Job\s*:", text)),
        "fit_header": bool(re.search(r"(?mi)^\s*Fit\s*:", text)),
        "region_header": bool(re.search(r"(?mi)^\s*Region\s*:", text)),
        "sample_header": bool(re.search(r"(?mi)^\s*Sample\s*:", text)),
        "hist_mode": bool(re.search(r"(?mi)^\s*ReadFrom\s*:\s*HIST\b", text)),
        "signal_sample": bool(re.search(r"(?mi)^\s*Type\s*:\s*SIGNAL\b", text)),
        "background_sample": bool(re.search(r"(?mi)^\s*Type\s*:\s*BACKGROUND\b", text)),
        "norm_factor": bool(re.search(r"(?mi)^\s*NormFactor\s*:", text)),
    }


def _append_diagnostic(solution_str: str, result: dict[str, Any]) -> None:
    """Append a compact per-rollout record when explicitly enabled.

    The host reward server processes requests serially, so appending one JSON
    line is safe and makes failures inspectable without retaining full model
    outputs or native workspaces.
    """
    value = os.environ.get("TREX_RL_REWARD_LOG")
    if not value:
        return
    path = Path(value)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "response_chars": len(solution_str),
            "response_excerpt": solution_str[:1000],
            "score": float(result.get("score", 0.0)),
            "reason": str(result.get("reason", "unknown")),
            "format_checks": _format_checks(solution_str),
            "structural_fraction": result.get("structural_fraction", 0.0),
            "native_success": int(result.get("native_success", 0)),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Diagnostics must never make a reward request fail.
        pass


def _extract_config(response: str) -> tuple[str | None, str | None]:
    """Accept raw config or one config fence, rejecting mixed prose."""
    text = response.strip()
    if not text:
        return None, "empty completion"
    fenced = re.fullmatch(r"```(?:text|config|ini)?\s*\n(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    if len(text) > MAX_CONFIG_CHARS:
        return None, f"completion exceeds {MAX_CONFIG_CHARS} characters"
    # Permit a model that prefaced an otherwise valid config with a short
    # sentence to reach structural scoring.  The task prompt still teaches
    # raw config-only output; this merely avoids an all-or-nothing signal.
    job = re.search(r'(?mi)^\s*Job\s*:\s*"?[A-Za-z_][A-Za-z0-9_]*"?', text)
    if job:
        text = text[job.start() :]
    else:
        return None, "completion has no Job block"
    return text + "\n", None


def _workspace_root() -> Path:
    configured = os.environ.get("TREX_RL_WORKSPACE_ROOT")
    root = Path(configured) if configured else PROJECT / "artifacts/trex_fitter/rl-rollouts"
    if not root.is_absolute():
        root = PROJECT / root
    root = root.resolve()
    try:
        root.relative_to(PROJECT)
    except ValueError as error:
        raise RuntimeError("TREX_RL_WORKSPACE_ROOT must be inside the repository") from error
    root.mkdir(parents=True, exist_ok=True)
    return root


def _reward_socket() -> Path | None:
    value = os.environ.get("TREX_RL_REWARD_SOCKET")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def _task_id(ground_truth: Any, extra_info: dict[str, Any] | None) -> str | None:
    for value in (ground_truth, extra_info):
        if isinstance(value, dict) and isinstance(value.get("task_id"), str):
            return value["task_id"]
    return None


def _score_locally(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Evaluate on the host that has access to podman-hpc."""
    del data_source
    def finish(result: dict[str, Any]) -> dict[str, Any]:
        _append_diagnostic(solution_str, result)
        return result

    if _task_id(ground_truth, extra_info) != "v3-histfit-001":
        return finish({"score": 0.0, "reason": "unknown task"})
    config_text, extraction_error = _extract_config(solution_str)
    if extraction_error:
        format_fraction = sum(_format_checks(solution_str).values()) / 8.0
        return finish({
            "score": round(FORMAT_REWARD_CAP * format_fraction, 6),
            "format_fraction": format_fraction,
            "native_success": 0,
            "reason": extraction_error,
        })

    keep_workspaces = os.environ.get("TREX_RL_KEEP_WORKSPACES", "0") == "1"
    timeout = int(os.environ.get("TREX_RL_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    workspace = Path(tempfile.mkdtemp(prefix="v3-histfit-001-", dir=_workspace_root()))
    try:
        v3_histfit_001.prepare(workspace)
        (workspace / "analysis.config").write_text(config_text, encoding="utf-8")
        result = v3_histfit_001.evaluate(workspace, execute=True, timeout=timeout)
        score = float(result.get("reward", 0.0))
        return finish({
            "score": score,
            "native_success": int(bool(result.get("success"))),
            "structural_fraction": sum(result.get("checks", {}).values()) / 8.0,
            "reason": str(result.get("reason", "unknown")),
        })
    except Exception as error:
        return finish({"score": 0.0, "native_success": 0, "reason": f"evaluator exception: {error}"})
    finally:
        if not keep_workspaces:
            shutil.rmtree(workspace, ignore_errors=True)


def _score_via_socket(socket_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Ask the host-side reward server to execute a native rollout."""
    timeout = int(os.environ.get("TREX_RL_REWARD_SECONDS", "2100"))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(socket_path))
        connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            chunk = connection.recv(65_536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk or sum(map(len, chunks)) > 65_536:
                break
    if not chunks:
        return {"score": 0.0, "reason": "empty host reward-server response"}
    return json.loads(b"".join(chunks).split(b"\n", 1)[0].decode("utf-8"))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """VERL entry point; delegate native work to host when configured."""
    socket_path = _reward_socket()
    if socket_path is not None:
        try:
            return _score_via_socket(socket_path, {
                "data_source": data_source, "solution_str": solution_str,
                "ground_truth": ground_truth, "extra_info": extra_info,
            })
        except Exception as error:
            return {"score": 0.0, "native_success": 0, "reason": f"reward-server error: {error}"}
    return _score_locally(data_source, solution_str, ground_truth, extra_info, **kwargs)


def main() -> None:
    """Local harness: score a config file with the native reward."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()
    if args.keep_workspace:
        os.environ["TREX_RL_KEEP_WORKSPACES"] = "1"
    answer = Path(args.config).read_text(encoding="utf-8")
    print(json.dumps(_score_locally("local", answer, {"task_id": "v3-histfit-001"}), indent=2))


if __name__ == "__main__":
    main()
