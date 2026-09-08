#!/usr/bin/env python3
"""Run a bundled TRExFitter config in the pinned Podman-HPC environment.

Examples:
  python3 trex_fitter/runner.py data/trex_config/hyy.config
  python3 trex_fitter/runner.py data/trex_config/FitExample.config --actions w f s
  python3 trex_fitter/runner.py --validate-all --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path


TREX_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TREX_DIR.parent
EXAMPLE_DIR = PROJECT_DIR / "data" / "trex_config"
INPUT_DIR = PROJECT_DIR / "data" / "trex_fitter" / "inputs"

sys.path.insert(0, str(TREX_DIR / "scripts"))
import trex as podman_trex  # noqa: E402


READ_FROM_RE = re.compile(r"^\s*ReadFrom:\s*(\S+)", re.MULTILINE)
MULTIFIT_RE = re.compile(r"^\s*MultiFit:\s*", re.MULTILINE)
DEFAULT_ACTIONS = {
    "NTUP": ["n", "w", "f", "s"],
    "HIST": ["h", "w", "f", "s"],
    "MULTIFIT": ["w", "f", "s"],
}


def config_path(value: str) -> Path:
    """Return a config under the project directory, suitable for /workdir."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path = path.resolve()
    if path.suffix != ".config" or not path.is_file():
        raise ValueError(f"Expected an existing .config file, got {path}")
    try:
        path.relative_to(PROJECT_DIR)
    except ValueError as exc:
        raise ValueError(
            f"{path} is outside {PROJECT_DIR}; it would not be mounted in the container."
        ) from exc
    return path


def read_from(path: Path) -> str:
    contents = path.read_text(errors="replace")
    match = READ_FROM_RE.search(contents)
    if not match:
        if MULTIFIT_RE.search(contents):
            return "MULTIFIT"
        raise ValueError(f"{path} does not declare Job: ReadFrom or MultiFit")
    return match.group(1).strip().strip('"').upper()


def default_actions(path: Path) -> list[str]:
    source = read_from(path)
    try:
        return DEFAULT_ACTIONS[source]
    except KeyError as exc:
        known = ", ".join(DEFAULT_ACTIONS)
        raise ValueError(
            f"{path} uses ReadFrom: {source}; choose actions explicitly with "
            f"--actions. Built-in defaults cover: {known}."
        ) from exc


def validate(path: Path) -> str:
    """Check the runner can map a config and its declared input mode."""
    source = read_from(path)
    container_path = podman_trex.to_container_path(path)
    return f"{path.relative_to(PROJECT_DIR)}: ReadFrom={source}, container={container_path}"


def container_cmd(command: str) -> list[str]:
    """Use host UID/GID ownership and mount inputs at /workdir/inputs."""
    if not INPUT_DIR.is_dir():
        raise RuntimeError(f"Missing input directory: {INPUT_DIR}")
    command_line = podman_trex.container_cmd(command)
    workdir_index = command_line.index("-w")
    command_line[workdir_index:workdir_index] = [
        "--userns=keep-id",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{INPUT_DIR}:/workdir/inputs:ro",
    ]
    return command_line


def check_setup() -> None:
    podman_trex.run(
        container_cmd(
            "which trex-fitter && "
            "test -d /workdir/inputs && "
            "test -d /workdir/inputs/test_inputs && "
            "echo 'TRExFitter and bundled inputs are available.'"
        ),
        label="check TRExFitter runner",
    )


def run_actions(path: Path, actions: list[str], log_dir: Path, compatibility_mode: bool) -> None:
    container_config = podman_trex.to_container_path(path)
    source = read_from(path)
    command_prefix = ""
    config_for_run = container_config
    container_log_dir = podman_trex.to_container_path(log_dir.resolve())
    if compatibility_mode:
        # UseGammaPulls was removed from the v1.10 Job schema.  Keep source
        # configs immutable and write a traceable, per-run copy instead.
        compatibility_dir = f"{container_log_dir}/compat-configs"
        config_for_run = f"{compatibility_dir}/{path.name}"
        command_prefix = (
            f"mkdir -p {shlex.quote(compatibility_dir)} && "
            f"sed '/^[[:space:]]*UseGammaPulls:/s/^/% /' {shlex.quote(container_config)} "
            f"> {shlex.quote(config_for_run)} && "
        )
    if source == "MULTIFIT":
        # The upstream MultiFit example refers to test/configs/*.config. Make
        # that conventional layout inside this run's persistent artifact
        # directory, with links to the bundled configs, before every action.
        container_work_dir = f"{container_log_dir}/multifit-work"
        container_config_dir = f"{container_work_dir}/test/configs"
        source_config_dir = "/workdir/data/trex_config"
        if compatibility_mode:
            command_prefix = (
                f"mkdir -p {shlex.quote(compatibility_dir)} && "
                f"for config in {source_config_dir}/*.config; do "
                f"sed '/^[[:space:]]*UseGammaPulls:/s/^/% /' \"$config\" "
                f"> {shlex.quote(compatibility_dir)}/\"$(basename \"$config\")\"; done && "
            )
            source_config_dir = compatibility_dir
        else:
            command_prefix = ""
        command_prefix += (
            f"mkdir -p {shlex.quote(container_config_dir)} && "
            f"ln -sfn {shlex.quote(source_config_dir)}/*.config {shlex.quote(container_config_dir)}/ && "
            f"cd {shlex.quote(container_work_dir)} && "
        )
    for action in actions:
        command = f"{command_prefix}trex-fitter {shlex.quote(action)} {shlex.quote(config_for_run)}"
        podman_trex.run(
            container_cmd(command),
            label=f"trex-fitter {action} {path.name}",
            log_dir=log_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="Config path relative to the repository root.")
    parser.add_argument(
        "--actions",
        nargs="+",
        help="TRExFitter actions to run. Defaults depend on the config's ReadFrom mode.",
    )
    parser.add_argument("--check", action="store_true", help="Check the Podman-HPC image and exit.")
    parser.add_argument(
        "--validate-all",
        action="store_true",
        help="Validate runner path handling for every data/trex_config/*.config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected actions without starting a container or changing analysis outputs.",
    )
    parser.add_argument(
        "--no-compatibility-mode",
        action="store_false",
        dest="compatibility_mode",
        default=True,
        help="Run the original config unchanged instead of adapting deprecated v1.10 settings.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for per-action stdout/stderr logs (default: artifacts/trex_fitter/<config name>).",
    )
    args = parser.parse_args()

    # trex.py normally assumes trex_fitter/ is /workdir. The example configs
    # instead live one level above it and refer to /workdir/inputs, so mount the
    # repository root consistently for every invocation.
    podman_trex.PROJECT_DIR = PROJECT_DIR

    if args.check:
        check_setup()

    if args.validate_all:
        for example in sorted(EXAMPLE_DIR.glob("*.config")):
            print(validate(example))

    if args.check or args.validate_all:
        if args.config is None:
            return

    if args.config is None:
        parser.error("config is required unless --check or --validate-all is used")

    try:
        path = config_path(args.config)
        actions = args.actions or default_actions(path)
    except ValueError as exc:
        parser.error(str(exc))

    if not actions:
        parser.error("at least one action is required")

    log_dir = args.log_dir or PROJECT_DIR / "artifacts" / "trex_fitter" / path.stem
    if args.dry_run:
        print(validate(path))
        print("actions:", " ".join(actions))
        print("log directory:", log_dir)
        return

    run_actions(path, actions, log_dir, args.compatibility_mode)


if __name__ == "__main__":
    main()
