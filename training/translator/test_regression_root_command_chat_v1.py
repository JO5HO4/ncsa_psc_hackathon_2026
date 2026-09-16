"""Proves the translator reproduces the committed qwen-root-command-chat-v1
dataset from the real ATLAS Open Data source, using (root_command, qwen3.5).

This is the migration proof for training/prepare_qwen_root_command_sft.py and
training/build_qwen_root_command_parquet.py, which this package replaces: the
committed dataset (already used for real training runs, see
results/ROOT_QWEN_CAMPAIGN.md) is never regenerated in place by this test --
it only proves the new, general code can reproduce it exactly.
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.translator import cli

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "data/datasets/atlas-open-data-sft-dataset/data/sft"
COMMITTED_DIR = REPO_ROOT / "data/derived/qwen-root-command-chat-v1"

SPLITS = {"train": 4480, "validation": 560}


@unittest.skipUnless(
    (SOURCE_DIR / "train.jsonl").is_file(),
    "atlas-open-data-sft-dataset submodule is not populated; run "
    "`git submodule update --init --recursive data/datasets/atlas-open-data-sft-dataset` first.",
)
class RootCommandRegressionTest(unittest.TestCase):
    def test_matches_committed_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for split, count in SPLITS.items():
                cli.run(
                    source=SOURCE_DIR / f"{split}.jsonl",
                    source_format="atlas-command",
                    task_name="root_command",
                    model_name="qwen3.5",
                    split=split,
                    output_dir=out,
                    repo_root=REPO_ROOT,
                    expected_count=count,
                )

                new_jsonl = (out / f"{split}.jsonl").read_bytes()
                committed_jsonl = (COMMITTED_DIR / f"{split}.jsonl").read_bytes()
                self.assertEqual(
                    new_jsonl, committed_jsonl,
                    f"{split}.jsonl is not byte-identical to the committed dataset",
                )

                new_df = pd.read_parquet(out / f"{split}.parquet").sort_values("id").reset_index(drop=True)
                old_df = pd.read_parquet(COMMITTED_DIR / f"{split}.parquet").sort_values("id").reset_index(drop=True)
                pd.testing.assert_frame_equal(new_df, old_df, check_like=True)


if __name__ == "__main__":
    unittest.main()
