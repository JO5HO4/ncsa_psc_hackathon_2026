#!/usr/bin/env python3
"""Materialize VERL GRPO records for the first native TRExFitter task.

The completion is the full text of ``analysis.config``. Native execution,
rather than a reference completion, provides the reward.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
TASK_ID = "v3-histfit-001"

PROMPT = """Create the complete contents of `analysis.config` for a native TRExFitter histogram fit.

Return only valid TRExFitter config text: no Markdown fences, no explanation, no shell command.

The pinned environment is StatAnalysis 0.8.2 / TRExFitter 1.10.0. The read-only
histogram fixture is mounted at `/workdir/inputs/examples/Histo`. It contains
the ROOT files `data.root`, `bkg1.root`, `bkg2.root`, and `sig.root`.

Build a signal-plus-background analysis with this known fixture layout:
- regions `SR_1` and `SR_2` use histogram `HTj`; `VR` uses `HTj_VR`;
- use a `Job` block with `ReadFrom: HIST`, `HistoPath: "/workdir/inputs/examples/Histo"`,
  and POI `SigXsecOverSM`;
- use a `Fit` block with `FitType: SPLUSB` and `FitRegion: CRSR`;
- create DATA `Data` from HistoFile `data`, BACKGROUND samples `Bkg1` and `Bkg2`
  from `bkg1` and `bkg2`, and SIGNAL `Signal` from `sig`;
- attach `NormFactor: "SigXsecOverSM",1,0,100` to the signal sample.

The config must run from a fresh directory using the native actions `h w f s`.
"""


def record(split: str, index: int) -> dict[str, object]:
    return {
        "data_source": "trex_fitter_rl",
        "prompt": [{"role": "user", "content": PROMPT}],
        "ability": "trex_config_generation",
        "reward_model": {"style": "rule", "ground_truth": {"task_id": TASK_ID}},
        "extra_info": {"task_id": TASK_ID, "split": split, "index": index},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(PROJECT / "artifacts/datasets/trex-v3-rl"))
    parser.add_argument("--train-records", type=int, default=32)
    parser.add_argument("--validation-records", type=int, default=4)
    args = parser.parse_args()
    if args.train_records < 1 or args.validation_records < 1:
        parser.error("both split sizes must be positive")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split, count, filename in (
        ("train", args.train_records, "train.parquet"),
        ("validation", args.validation_records, "validation.parquet"),
    ):
        pd.DataFrame([record(split, index) for index in range(count)]).to_parquet(output / filename, index=False)
    print(f"Wrote {args.train_records} train and {args.validation_records} validation records to {output}")


if __name__ == "__main__":
    main()
