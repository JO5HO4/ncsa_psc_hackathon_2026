"""Canonical authoring shape the translator consumes, and adapters into it.

``SourceRecord`` is the one shape every ``TaskProfile``/``ModelProfile``
combination renders from. Authoring a new dataset means producing a JSONL
file of ``{id, question, answer, category}`` records (see ``from_simple_qa``)
-- no message-wrapping, tools field, or model awareness required. Existing
pre-wrapped sources get a small adapter instead (see ``from_atlas_command_raw``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SourceRecord:
    id: str
    question: str
    answer: str
    category: Optional[str] = None
    # Present only when adapted from an existing pre-wrapped source: the full
    # original record (in its original key order) and its original user/
    # assistant message dicts, so a re-rendered row can stay byte-identical to
    # what was already built and trained on. Always ``None`` for a freshly
    # authored simple-QA record.
    raw: Optional[dict] = None
    user_message: Optional[dict] = None
    assistant_message: Optional[dict] = None


def from_simple_qa(record: dict) -> SourceRecord:
    """The primary authoring path: ``{id, question, answer, category}``.

    This is the format a new dataset (TRExFitter config, file-tasks, ...)
    should be authored in -- no pre-existing message wrapping or
    model-specific fields required, the translator adds those.
    """
    missing = [key for key in ("id", "question", "answer") if not record.get(key)]
    if missing:
        raise ValueError(f"Simple Q&A record missing required field(s): {missing}")
    return SourceRecord(
        id=record["id"],
        question=record["question"],
        answer=record["answer"],
        category=record.get("category"),
    )


def from_atlas_command_raw(record: dict) -> SourceRecord:
    """Adapter for the existing pre-wrapped ATLAS Open Data command-SFT shape
    (``data/datasets/atlas-open-data-sft-dataset/data/sft/*.jsonl``). Used for
    the ``root_command`` migration/regression case: ``raw`` and the original
    message dicts are carried through unmodified so translation only replaces
    the system prompt and tools/enable_thinking policy, exactly as the script
    this package replaces did.
    """
    identifier = record.get("id", "<unknown>")
    messages = record["messages"]
    roles = [message["role"] for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"{identifier}: expected system/user/assistant messages, found {roles}")
    answer = messages[2]["content"]
    if answer != record["ground_truth"]:
        raise ValueError(f"{identifier}: assistant message does not match ground_truth")
    return SourceRecord(
        id=record["id"],
        question=messages[1]["content"],
        answer=answer,
        category=record.get("api_family"),
        raw=dict(record),
        user_message=dict(messages[1]),
        assistant_message=dict(messages[2]),
    )


ADAPTERS = {
    "simple-qa": from_simple_qa,
    "atlas-command": from_atlas_command_raw,
}
