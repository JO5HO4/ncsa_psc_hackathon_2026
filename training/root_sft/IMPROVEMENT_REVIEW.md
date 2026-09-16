# Checkpoint comparison and ROOT-processing data improvement

## Current evidence and limits

Run: `artifacts/root-sft/full-3epochs-lr1e5-001`.
Model: Qwen/Qwen2.5-Coder-1.5B-Instruct, LoRA, unchanged verl.
Validation loss: epoch 1 = 2.9106, epoch 2 = 2.7810, epoch 3 = 2.7303.
The saved before/after outputs cover 46 validation questions. Earlier-epoch
generations are still needed before ranking all three checkpoints by correctness.
Do not substitute the separate one-epoch experiment for this run's epoch-1 checkpoint.

A qualitative inspection of epoch 3 identified these remediation priorities;
these are concrete observed failures, not a statistically established category ranking:

| Area | Example validation IDs | Error to correct | Correct teaching target |
|---|---|---|---|
| File inventory and cycles | 003, 005 | Keys confused with branches; cycles with loops | Inspect object classes and explicitly retrieve stored key versions. |
| Tree and collection semantics | 012, 042 | Rows confused with missing values; collection lengths with tree entries | Distinguish event rows from per-event object multiplicity. |
| Signed weights and histograms | 025, 045, 597 | Bin content treated as count; negative weights rejected or confused with underflow | Separate entries, sumw and sumw2; underflow depends on coordinates, not weight sign. |
| ROOT ownership | 400, 588 | Closing a file described as transferring ownership or resetting state | A directory-owned histogram may be deleted when its file closes; detach with SetDirectory(0) before closure when needed. |
| Fitting and likelihoods | 088, 092, 132, 511, 518 | Extended likelihood, profiling and fit-range option misidentified | Extended fits model yield as well as shape; profiling reoptimizes nuisances; R uses the function's fit range. |

The maintained reference answers for these failures already give the intended
corrections. Do not replace correct reference answers just because model outputs
are wrong. The new verified examples reinforce the file-processing corrections
with concrete executable counterexamples instead of copying validation questions.
Ownership and fitting still require a separate verified PyROOT/RooFit pack.

Sources: [ROOT files](https://root.cern/manual/root_files/),
[ownership](https://root.cern/manual/object_ownership/),
[TH1 fitting](https://root.cern/doc/master/classTH1.html),
[RooFit likelihood example](https://root.cern/doc/master/rf605__profilell_8py.html),
[Uproot guide](https://uproot.readthedocs.io/en/latest/basic.html).

## Run the comparison inside the existing GPU container

From `/workspace`, after the normal environment setup:

```bash
python training/root_sft/compare_checkpoints.py \
  --run artifacts/root-sft/full-3epochs-lr1e5-001 \
  --output artifacts/root-sft/checkpoint-comparison-001
```

This uses checkpoints 111 and 222 from this run, exports into a NEW directory,
and generates answers using the original system prompt, greedy decoding and 512
new-token limit. Base and epoch-3 predictions are reused after checksum checks.
Existing checkpoints, splits and predictions are not overwritten. Use a new
output suffix if a previous comparison attempt created the directory.

Progress/errors: `comparison.log`; process state: `status.json` (running,
failed subprocess, or complete). A forcibly killed process can leave running status.
Review `review.json` without opening `private_mapping.json` until scoring is done.
Score correctness 0 (wrong), 1 (partially correct), 2 (correct), flag any critical
error and give a short evidence-based note. Keep unreviewed scores null, not zero.
Report per-category counts and mean scores and an equal-category macro mean;
also report critical-error rates. Only 5–16 questions per category makes rankings
uncertain. This is assisted review, not expert certification or execution scoring.

## Verified training-candidate pack

```bash
python training/root_sft/verified_examples.py \
  --output data/root_io/verified/root-processing-v1
python -m unittest training.root_sft.test_improvements -v
```

Requires Uproot 5, Awkward 2 and NumPy. The builder writes a temporary synthetic
TTree, runs each authored answer, checks exact expected outputs, and cleans up the
temporary ROOT files. It refuses an existing output directory. Published snippets
can be run against the fixture created by `fixture(path)` in the same module.
Never run untrusted generated model code through this verifier.

The pack has 12 examples, exactly two per task type: inventory, selection, jagged
arrays, weighted histograms, streaming/schema validation, and writing/reopening.
`examples.jsonl` contains Q&A, runnable code, expected outputs, messages, sources,
and a shared split group. `verification.json` records package versions, data hash
and counts. It is synthetic training material, not a new held-out benchmark and
not expert-certified coverage of ROOT. No automatic training launch occurs.

## Balance policy for the next training release

Current train counts: files_io 50, trees_analysis 28, histograms_plotting 22,
hep_interpretation 38, statistics 84. All 222 examples are Q&A, with 192 labeled
grounded knowledge and 30 contextual analysis. Topic caps alone do NOT produce
equal task-type exposure. The new pack is balanced internally; appending it would
NOT make the whole existing dataset balanced.

Before the next training run:

1. Label examples by both subject and operation: explain, inspect, transform,
   debug, validate, and write. Audit answer lengths/tokens and difficulty too.
2. For the ROOT-processing track, target equal shares across the six executable
   task families above. Maintain a separately reported knowledge track so an
   abundance of statistics prose cannot dominate file-processing practice.
3. Select unique verified examples to meet quotas; report unfilled cells instead
   of duplicating answers to fill them. Never balance by moving held-out rows.
4. Keep this entire shared-fixture pack in training only. Design independently
   grouped evaluation fixtures with different schemas and edge cases. Because
   validation failures informed remediation, further validation gains are
   development feedback, not independent evidence of generalization.
5. Version the new release and report row AND assistant-token shares by task.
   Retain existing group-leakage and truncation checks. Do not mix these records
   into the frozen v1 release in place. Convert the finalized mixture to the
   same messages/Parquet format used by the existing verl adapter.

Equal counts reduce one source of imbalance; they do not guarantee an unbiased
model. The final test remains sealed until data and hyperparameters are fixed.
