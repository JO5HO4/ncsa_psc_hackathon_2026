"""Small VERL adapter for the checked-in config SFT reference dataset."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer.chat_template import apply_chat_template, extract_system_prompt_and_generation


class TReXConfigSFTDataset(MultiTurnSFTDataset):
    """Load nested SFT records safely and decode their JSON-encoded tools."""

    @staticmethod
    def decode_tools(value):
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, list):
            raise TypeError(
                "The tools column must contain a JSON array or Python list; "
                f"received {type(value).__name__}."
            )
        return value

    def _read_files_and_process(self):
        # VERL's default ``dtype_backend="pyarrow"`` loader aborts when Pandas
        # indexes nested struct/list columns in the ATLAS command Parquet files.
        # Default Pandas objects preserve those values and support safe ``iloc``
        # access in the StatefulDataLoader.
        frames = [pd.read_parquet(path) for path in self.parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)

        total = len(self.dataframe)
        print(f"dataset len: {total}")
        if 0 < self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(self.seed) if self.seed is not None else np.random.default_rng()
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()].reset_index(drop=True)
            print(f"selected {self.max_samples} random samples out of {total}")

        self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()
        if self.tools_key in self.dataframe.columns:
            self.tools = [self.decode_tools(value) for value in self.dataframe[self.tools_key].tolist()]
        else:
            self.tools = None
        self.enable_thinking = (
            self.dataframe[self.enable_thinking_key].tolist()
            if self.enable_thinking_key in self.dataframe.columns
            else None
        )
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(
            self.tokenizer, **self.apply_chat_template_kwargs
        )

    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        """Render each turn with enough preceding context for Qwen's template."""
        # Qwen 3.5 rejects an isolated system message. It is included in the
        # following user/assistant contexts, so it needs no standalone tokens.
        if message["role"] == "system":
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty.clone(), empty.clone(), {}

        processor = self.processor if self.processor is not None else self.tokenizer
        kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking

        context = full_message[: index + 1]
        inputs = dict(
            apply_chat_template(
                processor,
                messages=context,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **kwargs,
            )
        )
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]

        # Keep only tokens introduced by this turn. The first user turn needs
        # the system context, while assistant and later user turns can subtract
        # their valid preceding conversation.
        prefix = full_message[:index]
        if any(turn["role"] == "user" for turn in prefix):
            prefix_inputs = dict(
                apply_chat_template(
                    processor,
                    messages=prefix,
                    tools=tools,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    **kwargs,
                )
            )
            prefix_length = prefix_inputs["input_ids"].shape[1]
            input_ids = input_ids[prefix_length:]
            attention_mask = attention_mask[prefix_length:]

        loss_mask = torch.ones_like(attention_mask) if message["role"] == "assistant" else torch.zeros_like(attention_mask)
        if message["role"] == "assistant":
            loss_mask[: len(self.generation_prompt)] = 0
        return input_ids, loss_mask, attention_mask, inputs
