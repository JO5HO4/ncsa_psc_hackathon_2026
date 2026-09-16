"""Target-model rendering quirks: what to put in the tools/enable_thinking columns.

These are the two levers that turned out to matter for Qwen3.5's chat template
(see README.md for the full story): the ``tools`` field selects between its
plain-chat and XML function-calling template branches, and ``enable_thinking``
toggles its reasoning-trace branch. Neither is task content -- they are purely
about which of the model's own template branches gets used at tokenization
time (owned by verl/verl_dataset.py), so they belong to the model, not to any
one dataset.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    tools_value: str
    enable_thinking: bool
    notes: str


QWEN3_5 = ModelProfile(
    name="qwen3.5",
    tools_value="[]",
    enable_thinking=False,
    notes=(
        "Empty tools deliberately selects Qwen's plain chat template branch "
        "instead of its XML function-calling branch. enable_thinking=False "
        "disables the <think> reasoning-trace branch so a response begins "
        "directly with the requested answer."
    ),
)

MODEL_PROFILES = {QWEN3_5.name: QWEN3_5}


def get_model_profile(name: str) -> ModelProfile:
    try:
        return MODEL_PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown model profile {name!r}; known: {sorted(MODEL_PROFILES)}") from None
