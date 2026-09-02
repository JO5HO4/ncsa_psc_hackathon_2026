"""Small VERL adapter for the checked-in config SFT reference dataset."""

from __future__ import annotations

import json

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class TReXConfigSFTDataset(MultiTurnSFTDataset):
    """Decode the JSON-encoded ``tools`` column before VERL renders messages."""

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
        super()._read_files_and_process()
        if self.tools is not None:
            self.tools = [self.decode_tools(value) for value in self.tools]
