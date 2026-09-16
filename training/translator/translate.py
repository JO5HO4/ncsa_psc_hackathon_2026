"""Combine a SourceRecord with a TaskProfile and ModelProfile into one training row.

This is the translator's core: it never talks to a tokenizer or a real model
(that stays owned by ``verl``/``training/verl_dataset.py``) -- it only decides
what the ``messages``/``tools``/``enable_thinking`` columns should contain.
"""
from __future__ import annotations

from .model_profiles import ModelProfile
from .records import SourceRecord
from .task_profiles import TaskProfile


def translate_record(source: SourceRecord, task: TaskProfile, model: ModelProfile) -> dict:
    task.validate_answer(source.answer)

    system_message = {"role": "system", "content": task.system_prompt, "tool_calls": None, "tool_call_id": None}
    user_message = source.user_message or {
        "role": "user", "content": source.question, "tool_calls": None, "tool_call_id": None,
    }
    assistant_message = source.assistant_message or {
        "role": "assistant", "content": source.answer, "tool_calls": None, "tool_call_id": None,
    }
    messages = [system_message, user_message, assistant_message]

    # Start from the original record when one exists, so its pre-existing
    # keys keep their original position and only tools/enable_thinking/
    # format_version are appended new -- this is what makes re-rendering an
    # already-published dataset byte-identical to what was actually trained on.
    if source.raw is not None:
        row = dict(source.raw)
    else:
        row = {"id": source.id}
        if source.category is not None:
            row["category"] = source.category
    row["id"] = source.id
    row["messages"] = messages
    row["tools"] = model.tools_value
    row["enable_thinking"] = model.enable_thinking
    row["format_version"] = task.format_version
    return row


def render_split(records: list, task: TaskProfile, model: ModelProfile) -> list:
    seen = set()
    rows = []
    for source in records:
        if source.id in seen:
            raise ValueError(f"Duplicate id: {source.id}")
        seen.add(source.id)
        rows.append(translate_record(source, task, model))
    return rows
