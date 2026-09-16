"""Dataset-family output contracts: the exact shape a valid assistant answer must have.

A TaskProfile owns the system prompt that states the contract and a validator
that checks a candidate answer against it. This is model-agnostic -- the same
contract applies no matter which model is being trained on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TaskProfile:
    name: str
    system_prompt: str
    validate_answer: Callable[[str], None]
    format_version: str


def _validate_root_command(answer: str) -> None:
    if "\n" in answer:
        raise ValueError("ROOT command answer must be a single line")
    if not answer.startswith("root -l -b -q -e '") or not answer.endswith("'"):
        raise ValueError("ROOT command answer must be wrapped in root -l -b -q -e '...'")


# Migrated verbatim from training/prepare_qwen_root_command_sft.py, which this
# package replaces. format_version is kept identical so existing consumers of
# data/derived/qwen-root-command-chat-v1 see no change.
ROOT_COMMAND = TaskProfile(
    name="root_command",
    system_prompt=(
        "Return exactly one executable ROOT command on one line. "
        "The entire response must begin with root -l -b -q -e ' and end with one matching single quote. "
        "Inside it, print exactly one RESULT=<value> line and call gSystem->Exit(0). "
        "Do not add Markdown, backticks, explanations, XML, JSON, tool calls, or any other text."
    ),
    validate_answer=_validate_root_command,
    format_version="qwen-root-command-chat-v1",
)

TASK_PROFILES = {ROOT_COMMAND.name: ROOT_COMMAND}


def get_task_profile(name: str) -> TaskProfile:
    try:
        return TASK_PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown task profile {name!r}; known: {sorted(TASK_PROFILES)}") from None
