"""Small VERL adapter for the checked-in config SFT reference dataset."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer.chat_template import extract_system_prompt_and_generation


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
