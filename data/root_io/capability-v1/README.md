# ROOT file-handling baseline: ten questions

Evaluation only. Do not add these prompts or answers to training and later call
their scores held-out performance. No training framework changes are made.

The saved questions cover five families, two per family: discovery/safety,
reading/extraction, collections/streaming, histograms/ownership, and multi-file
validation/writing. Each asks for a reusable Python function, not an invented
numerical result from a filename.

The real reference is `data/samples/examples/Ntuple/tt_aMCNloP8.root`, inspected
read-only. It contains TTree `nominal_Loose`, 6632 entries and 15 scalar branches.
It has no stored histograms or jagged branches. `sample_metadata.json` records
the observed object/branch metadata and file hash.

`fixture.root` and `incompatible.root` are explicitly synthetic supplements for
histograms, repeated identifiers, empty/jagged collections, a count mismatch,
nonfinite scalar values and incompatible schema. They are not observations from
the ttbar sample. Their types and file hashes were checked during preparation.

## Run inside the existing GPU container

From `/workspace`, with exactly one visible GPU:

```bash
python -u training/root_sft/root_capability.py run \
  --directory /workspace/artifacts/root-sft/qwen-root-capability-10-001
```

Uses the original `Qwen/Qwen2.5-Coder-1.5B-Instruct`, pinned revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a`, unchanged tutor system prompt, greedy
decoding and 1024 new tokens, matching the earlier inference settings.
One response per question; no retries, adapters or weight updates.

This is code generation with supplied metadata, **not tool-assisted file
analysis**. The model is not given reference solutions or fixture row values.
Generated code is saved as text and never executed. Run it only after review
in an isolated execution environment with read-only inputs and scratch outputs.
The current pack has no verified reference implementations or automatic code
correctness grader; those remain required before reporting execution accuracy.
The 1024-token cap can truncate longer answers; this is a measurement limitation.
Do not compare its score directly with the earlier conceptual 50-question test.

Outputs: `inference.log`, `runtime.json`, `status.json`, `answers.jsonl`,
`ANSWERS.md`, and `review.json` with initially unset correctness grades.
Existing run directories are refused to protect previous results.

Preparation was performed with `python training/root_sft/root_capability.py
prepare`. It refuses to replace a frozen manifest; no preparation is needed
inside the GPU container. The real file need not be mounted there because its
metadata is already frozen in the prompts and no generated code is executed.
