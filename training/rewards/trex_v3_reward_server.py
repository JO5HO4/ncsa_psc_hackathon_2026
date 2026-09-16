#!/usr/bin/env python3
"""Host-side serialized native reward service for VERL TRExFitter GRPO.

Run this on the interactive allocation host, outside the VERL container. The
container connects over a Unix socket in the repository bind mount; this keeps
podman-hpc on the host where it is supported.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import socket
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from training.rewards import trex_v3_histfit_reward as reward  # noqa: E402


def resolve_socket(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT / path
    path = path.resolve()
    try:
        path.relative_to(PROJECT)
    except ValueError as error:
        raise ValueError("socket must be inside the repository") from error
    return path


def respond(connection: socket.socket, message: dict[str, object]) -> None:
    connection.sendall((json.dumps(message, allow_nan=False) + "\n").encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, help="relative-to-repository Unix socket path")
    parser.add_argument("--timeout", type=int, default=600, help="native fit timeout per rollout")
    args = parser.parse_args()
    path = resolve_socket(args.socket)
    path.parent.mkdir(parents=True, exist_ok=True)
    # On the CFS Lustre mount, stat()/exists() on a stale Unix socket can
    # itself fail with EINVAL. Unlinking this exact user-configured path is
    # reliable, including for a stale socket from an interrupted allocation.
    try:
        path.unlink()
    except FileNotFoundError:
        pass

    os.environ.pop("TREX_RL_REWARD_SOCKET", None)
    os.environ["TREX_RL_TIMEOUT_SECONDS"] = str(args.timeout)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        # Some Lustre mounts reject chmod on Unix-domain sockets (EINVAL).
        # The process umask still protects the newly-created socket there;
        # enforce 0600 wherever the mount supports it.
        try:
            path.chmod(0o600)
        except OSError as error:
            if error.errno != errno.EINVAL:
                raise
        server.listen(16)
        print(f"TREX_RL_REWARD_SERVER_READY socket={path}", flush=True)
        try:
            while True:
                connection, _ = server.accept()
                with connection:
                    payload = connection.recv(65_536)
                    try:
                        request = json.loads(payload.split(b"\n", 1)[0].decode("utf-8"))
                        result = reward._score_locally(
                            str(request.get("data_source", "")),
                            str(request.get("solution_str", "")),
                            request.get("ground_truth"),
                            request.get("extra_info"),
                        )
                    except Exception as error:
                        result = {"score": 0.0, "native_success": 0, "reason": f"server exception: {error}"}
                    respond(connection, result)
        finally:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
