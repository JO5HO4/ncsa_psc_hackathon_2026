"""Reusable container and subprocess helpers; no command-line entry point."""

from __future__ import annotations

import os
import math
import re
import shlex
import subprocess
import sys
import codecs
import selectors
from dataclasses import dataclass
from typing import Callable, TextIO
import signal
import time
from pathlib import Path


# StatAnalysis 0.8.2; digest resolved from the tested registry image.
IMAGE = "gitlab-registry.cern.ch/atlas/statanalysis@sha256:36a8c06ae90401e3629830c8ffe3b9fcf0b2a1841b28f59e4ecfa49c243051e1"
CONTAINER_ENGINE = "podman-hpc"

PROJECT_DIR = Path(os.environ.get("ATLASRL_PROJECT_DIR", Path(__file__).resolve().parent)).resolve()
CONTAINER_PROJECT_DIR = "/workdir"


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def to_container_path(path: str | Path) -> str:
    """
    Convert a project-local path to the corresponding path inside the container.

    Example:
      configs/myfit.config -> /workdir/configs/myfit.config
    """
    path = Path(path)

    if path.is_absolute():
        path = path.resolve()
        try:
            rel = path.relative_to(PROJECT_DIR)
        except ValueError as exc:
            raise ValueError(
                f"Path {path} is outside PROJECT_DIR={PROJECT_DIR}. "
                "Move it under the project directory or add another mount."
            ) from exc
    else:
        rel = path

    return str(Path(CONTAINER_PROJECT_DIR) / rel)

def discover_symlink_target_mounts() -> list[tuple[str, str]]:
    """
    Find symlinks inside PROJECT_DIR whose targets live outside PROJECT_DIR.

    Mount each external target at the same absolute path inside the container,
    so absolute symlinks continue to work:

        inputs/Data -> /global/.../gamgam_data

    becomes usable because /global/.../gamgam_data is mounted into the container.
    """
    mounts: set[tuple[str, str]] = set()
    project = PROJECT_DIR.resolve()

    for link in PROJECT_DIR.rglob("*"):
        if not link.is_symlink():
            continue

        raw_target = os.readlink(link)
        literal_target = Path(raw_target)
        if not literal_target.is_absolute():
            literal_target = (link.parent / literal_target).resolve()
        resolved_target = link.resolve()

        if not resolved_target.exists():
            continue

        try:
            resolved_target.relative_to(project)
            continue  # target is already inside project mount
        except ValueError:
            pass

        for target in {literal_target, resolved_target}:
            mount_path = target if resolved_target.is_dir() else target.parent
            mounts.add((str(mount_path), str(mount_path)))

    return sorted(mounts)


def container_cmd(
    shell_command: str, *, discover_external_mounts: bool = True
) -> list[str]:
    """
    Build a non-interactive podman-hpc command for TRExFitter.

    - no -it from Python
    - project is mounted as /workdir
    - external symlink targets are mounted at their original absolute paths
    - --group-add keep-groups preserves NERSC project group permissions
    """
    mounts = [(str(PROJECT_DIR), CONTAINER_PROJECT_DIR, "rw")]

    if discover_external_mounts:
        for src, dst in discover_symlink_target_mounts():
            mounts.append((src, dst, "ro"))

    cmd = [
        CONTAINER_ENGINE,
        "run",
        "--rm",
        "--group-add",
        "keep-groups",
        "--env",
        "HOME=/tmp",
        "--env",
        "XDG_CACHE_HOME=/tmp/.cache",
    ]

    seen = set()
    for src, dst, mode in mounts:
        key = (src, dst)
        if key in seen:
            continue
        seen.add(key)
        cmd.extend(["-v", f"{src}:{dst}:{mode}"])

    cmd.extend(
        [
            "-w",
            CONTAINER_PROJECT_DIR,
            IMAGE,
            "bash",
            "-lc",
            f"umask 0002; {shell_command}",
        ]
    )

    return cmd


@dataclass
class RunResult:
    returncode: int
    timed_out: bool
    stdout: str
    stderr: str
    elapsed_seconds: float
    cleanup_error: str | None = None


def run(
    cmd: list[str], label: str, log_dir: Path | None = None, *,
    timeout: float | None = None, start_new_session: bool = False,
    check: bool = True, output_stream: TextIO | None = None,
    cleanup: Callable[[], None] | None = None,
) -> RunResult:
    """Stream and save this invocation's output, optionally bounding its lifetime.

    A timed command owns a new POSIX process group so its workers are terminated
    too. The cleanup callback handles resources outside that group (containers).
    """
    if timeout is not None and not start_new_session:
        raise ValueError("A timeout requires an owned process group")
    destination = output_stream if output_stream is not None else sys.stdout
    print(f"\n=== {label} ===", file=destination, flush=True)
    print(quote_cmd(cmd), file=destination, flush=True)
    started = time.monotonic()
    stdout: list[str] = []
    stderr: list[str] = []
    timed_out = False
    cleanup_error = None
    log_files = []
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_label = label.replace(" ", "_").replace("/", "_")
        log_files = [
            (log_dir / f"{safe_label}.{stream}.log").open("w")
            for stream in ("stdout", "stderr")
        ]

    def stop_group(process):
        nonlocal cleanup_error
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The group leader may exit before a worker that ignores SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        if cleanup is not None:
            try:
                cleanup()
            except Exception as error:
                cleanup_error = str(error)

    process = None
    selector = selectors.DefaultSelector()
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, start_new_session=start_new_session,
        )
        for index, (pipe, target, collected) in enumerate((
            (process.stdout, destination, stdout), (process.stderr, sys.stderr, stderr)
        )):
            selector.register(pipe, selectors.EVENT_READ, (
                target, collected, log_files[index] if log_files else None,
                codecs.getincrementaldecoder("utf-8")(errors="replace"),
            ))
        drain_deadline = None
        while selector.get_map() or process.poll() is None:
            now = time.monotonic()
            if timeout is not None and not timed_out and now - started >= timeout:
                timed_out = True
                stop_group(process)
                # Do not hang on an escaped process holding pipes if external
                # cleanup failed; preserve the diagnostic and partial logs.
                drain_deadline = time.monotonic() + 1
            if drain_deadline is not None and time.monotonic() >= drain_deadline:
                break
            for key, _ in selector.select(timeout=0.05):
                target, collected, log, decoder = key.data
                chunk = os.read(key.fileobj.fileno(), 65536)
                text = decoder.decode(chunk, final=not chunk)
                if text:
                    collected.append(text)
                    if log is not None:
                        log.write(text)
                        log.flush()
                    print(text, end="", file=target, flush=True)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
        process.wait()
    except BaseException:
        if process is not None and process.poll() is None:
            if start_new_session:
                stop_group(process)
            else:
                process.terminate()
                process.wait()
        raise
    finally:
        selector.close()
        if process is not None:
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        for log in log_files:
            log.close()
    result = RunResult(
        process.returncode, timed_out, "".join(stdout), "".join(stderr),
        time.monotonic() - started, cleanup_error,
    )
    if check and (result.returncode != 0 or result.timed_out):
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def remove_container(cid_file: Path) -> None:
    """Remove only the container identified by this invocation's CID file."""
    if not cid_file.is_file():
        return
    container_id = cid_file.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
        raise ValueError(f"Invalid container ID in {cid_file}")
    result = subprocess.run(
        [CONTAINER_ENGINE, "rm", "--force", "--ignore", container_id],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode:
        raise RuntimeError(f"Container cleanup failed: {result.stderr.strip()}")


def parse_significance_from_text(text: str) -> float | None:
    patterns = [
        r"SIGNIFICANCE\s*=\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        r"Observed\s+significance(?:\s+mu\s*=\s*\S+)?(?:\s*\([^)]*\))?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        r"obs(?:erved)?\s+significance(?:\s+mu\s*=\s*\S+)?(?:\s*\([^)]*\))?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        r"Significance(?:\s+mu\s*=\s*\S+)?(?:\s*\([^)]*\))?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        r"Z0\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return value if math.isfinite(value) else None
    return None
