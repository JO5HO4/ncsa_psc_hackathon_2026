"""Dependency-free interface for evaluating a TRExFitter configuration.

The caller supplies a finished ``.config`` file. This module owns runner
selection, workspace logs, timeouts, and significance extraction; it never
accepts shell commands from a model or agent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ACTIONS = ("n", "w", "f", "s")
_SIGNIFICANCE_PATTERNS = (
    r"SIGNIFICANCE\s*=\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    r"Observed\s+significance(?:\s+mu\s*=\s*\S+)?(?:\s*\([^)]*\))?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    r"Significance(?:\s+mu\s*=\s*\S+)?(?:\s*[:=])\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    r"Z0\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
)


@dataclass(frozen=True)
class TrexResult:
    """Structured outcome of one evaluator-owned TRExFitter invocation."""

    config_path: str
    log_dir: str
    actions: tuple[str, ...]
    success: bool
    returncode: int
    significance: float | None
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TrexFitterInterface:
    """Run a final project-local config with the real or mock backend."""

    def __init__(
        self,
        project_dir: str | Path | None = None,
        runner_path: str | Path | None = None,
        timeout: int = 1200,
    ) -> None:
        self.project_dir = Path(project_dir or Path(__file__).parent).resolve()
        self.runner_path = Path(runner_path or self.project_dir / "scripts" / "trex.py").resolve()
        self.timeout = timeout

    @classmethod
    def mock(cls, **kwargs: object) -> "TrexFitterInterface":
        project_dir = Path(kwargs.pop("project_dir", Path(__file__).parent)).resolve()
        return cls(
            project_dir=project_dir,
            runner_path=project_dir / "scripts" / "mock_trex.py",
            **kwargs,
        )

    def run(
        self,
        config_path: str | Path,
        *,
        actions: Iterable[str] = DEFAULT_ACTIONS,
        log_dir: str | Path | None = None,
    ) -> TrexResult:
        """Run a completed config and return its parsed significance.

        The source config may live in the dataset. The interface copies it into
        its private work folder before asking the container runner to use it.
        ``actions`` is evaluator configuration, never model output.
        """

        source_config = Path(config_path).resolve()
        if source_config.suffix != ".config":
            raise ValueError(f"Expected a .config file, got {source_config}")
        if not source_config.is_file():
            raise FileNotFoundError(source_config)
        if not self.runner_path.is_file():
            raise FileNotFoundError(self.runner_path)

        selected_actions = tuple(actions)
        if not selected_actions:
            raise ValueError("At least one TRExFitter action is required")

        output_dir = Path(log_dir or self._default_log_dir()).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        config = output_dir / "input.config"
        shutil.copy2(source_config, config)
        command = [
            sys.executable,
            str(self.runner_path),
            str(config),
            "--project-dir",
            str(self.project_dir),
            "--log-dir",
            str(output_dir),
            "--actions",
            *selected_actions,
        ]

        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )
            returncode = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            returncode = -999
            stdout = exc.stdout or ""
            stderr = exc.stderr or f"Timed out after {self.timeout} seconds."

        significance = _read_significance(output_dir, stdout, stderr)
        return TrexResult(
            config_path=str(source_config),
            log_dir=str(output_dir),
            actions=selected_actions,
            success=returncode == 0 and significance is not None,
            returncode=returncode,
            significance=significance,
            stdout=stdout,
            stderr=stderr,
        )

    def _default_log_dir(self) -> Path:
        return self.project_dir / "workspaces" / "evaluations" / uuid.uuid4().hex


def _read_significance(log_dir: Path, stdout: str, stderr: str) -> float | None:
    result_path = log_dir / "results.json"
    if result_path.is_file():
        try:
            return float(json.loads(result_path.read_text(encoding="utf-8"))["significance"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    text = f"{stdout}\n{stderr}"
    for pattern in _SIGNIFICANCE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None
