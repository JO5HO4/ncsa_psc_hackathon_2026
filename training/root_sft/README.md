# ROOT knowledge post-training with the existing verl framework

The separate [executable file-operation pilot](FILE_TASKS.md) adds balanced
structured-operation SFT data and a restricted execution evaluator. It does not
modify this knowledge-bank release or launch training automatically.

For the three-epoch checkpoint comparison, observed failure modes, executable
training candidates, and the next release's task-balance policy, see
[IMPROVEMENT_REVIEW.md](IMPROVEMENT_REVIEW.md).

This pipeline uses `Qwen/Qwen2.5-Coder-1.5B-Instruct` and the repository's unchanged `training/scripts/run_verl_sft.sh`, dataset adapter, model exporter and inference runner. It adds data preparation, preflight checks, Perlmutter job configuration and paired evaluation; it does not introduce another trainer or change verl.

The first experiment is supervised fine-tuning (SFT). LoRA trains low-rank adapters while freezing the original base parameters. Inference merges the learned delta into the effective model weights. It is real weight adaptation, not prompting or retrieval, but it is not full-parameter fine-tuning. The input model is already instruction-tuned; “baseline” means before our ROOT-specific training.

## 1. Prepare and inspect the dataset

From the repository root, in an environment with Python 3.10+:

```bash
python -m pip install -r training/root_sft/requirements.txt
python training/root_sft/prepare.py \
  --allow-draft \
  --output data/root_io/splits/root-sft-v1
python -m unittest training.root_sft.test_pipeline -v
```

The current release is already generated in this workspace. The builder deliberately refuses to overwrite a nonempty output directory; use a new version directory when rebuilding. Generated splits are gitignored, so rebuild them after cloning on another machine. `root_questions_617.jsonl` and the student release are not modified. `--allow-draft` explicitly acknowledges that the bank has not received independent expert certification; it does not promote its review status.

Each Parquet record has system/user/assistant `messages`, an empty JSON-encoded `tools` list accepted by the existing adapter, and tracing metadata. The user turn includes the question and its supplied hypothetical context, but not the answer, required facts or grading rubric. The assistant turn contains only the reference answer. Tokenization preflight uses the real verl adapter and checks every record's token limit, chat-template equivalence and assistant-only loss mask.

### Selection and balance

The default is one or two distinct-answer examples per canonical topic, preferring difficulty/kind diversity within that topic. There is no oversampling or invented content. All 617 IDs remain in `assignments.jsonl`, including excluded rows and their reasons. The seed and source/artifact hashes are recorded in `manifest.json`.

| Category | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Files and I/O | 50 | 11 | 9 |
| Trees and analysis | 28 | 6 | 6 |
| Histograms and plotting | 22 | 5 | 5 |
| Statistics | 84 | 16 | 18 |
| HEP interpretation | 38 | 8 | 8 |
| Total | **222** | **46** | **46** |

Training difficulty counts are 60 introductory, 97 intermediate and 65 advanced. Every split covers all five categories and all three difficulties. This is **topic-balanced, not equal-category sampling**: the source contains more distinct statistics concepts. Evaluation reports category means and an equal-category macro average so a dominant category cannot hide weaker ones.

314 examples are selected: 241 additional variants are excluded by the two-per-topic cap and 62 by exact normalized-answer deduplication. All source examples are considered; not all are used as optimizer targets. Change `--topic-cap` only as a new, separately evaluated dataset version.

### Split policy and limitations

Before selection, form connected groups using canonical topics (`.b` shares its parent's topic), original scenario/split groups, normalized exact questions and normalized exact answers. Assign whole connected groups toward 70/15/15 proportions, balancing category, difficulty and question kind. Legacy split labels are retained only in the audit, not reused for training.

The repeated hypothetical scenarios connect into one large source group. Its 30 selected examples end up in training; all 46 validation and 46 test examples are grounded-knowledge questions. Therefore this experiment measures unseen-topic knowledge answers, **not held-out contextual tool execution**. Known linked examples do not cross splits, but this is not a guarantee against every semantic similarity or against a model having seen public material during pretraining. Independent executable ROOT evaluation remains future work.

## 2. Configure a Perlmutter GPU run

The supplied batch script requests one GPU, one task and 32 logical CPUs in the shared QOS, following [NERSC's single-GPU example](https://docs.nersc.gov/systems/perlmutter/running-jobs/). It starts a noninteractive [podman-hpc GPU container](https://docs.nersc.gov/development/containers/podman-hpc/overview/), then uses the existing setup and SFT scripts.

Before submitting, download the same image used by the repository's training container:

```bash
podman-hpc pull docker.io/verlai/verl:sgl059.latest
podman-hpc image inspect docker.io/verlai/verl:sgl059.latest --format '{{.RepoDigests}}'
```

For repeated experiments, set `VERL_IMAGE` to the returned immutable `docker.io/verlai/verl@sha256:...` reference. The wrapper records the actual image ID as well. The image was **not present** in the inspected login environment; it has not been downloaded automatically.

The model defaults to the official [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct). Preflight resolves one Hub snapshot and uses that exact local snapshot for baseline inference and training. Set `MODEL_REVISION` to a Hub commit hash to reproduce it across runs. Initial model download and session dependency installation require network access and count toward job time. Cache is node-local `/tmp`; it is not a durable copy across jobs. Preserve a snapshot separately if your site's compute-node network policy requires pre-staging.

Run commands from the repository root. Replace `YOUR_GPU_ACCOUNT` with your authorized NERSC GPU allocation; it cannot be inferred from the directory name. No job is submitted merely by preparing the data.

## 3. Run the two-step GPU smoke test first

```bash
MODE=smoke sbatch --account=YOUR_GPU_ACCOUNT training/root_sft/perlmutter.sbatch
```

The smoke job:

1. Verifies dataset hashes, CUDA access, all tokenized examples and loss masks.
2. Generates baseline answers for the first four validation prompts.
3. Runs **two optimizer steps** through the existing verl SFT launcher.
4. Exports the checkpoint and verifies a finite, nonzero effective LoRA weight delta.
5. Reloads the export, generates answers for the identical prompts and prepares blinded paired review.

Defaults: one GPU, global batch 2, microbatch 1, maximum sequence length 2048, LoRA rank/alpha 16, learning rate `1e-5`, no automatic resume, and torch compilation disabled for the small run. Batch 2 divides both train and validation counts; preflight rejects settings that would silently drop an incomplete batch in the current verl loader. Sequence truncation and tokenization mismatch are errors, not silently ignored.

The wrapper disables the launcher's inline export and calls the existing `inference/export_verl_checkpoint.sh` separately. This safely handles verl's newline-free checkpoint tracker, which otherwise makes the existing launcher's plain `read` exit under `set -e`. The underlying launcher, exporter and training framework remain unchanged.

Read `root-sft-JOBID.out` and `artifacts/root-sft/smoke-JOBID/`. Success requires `preflight.json`, `weight_update.json`, an exported checkpoint, both prediction files, and `review/paired_review.jsonl`. A nonzero weight delta is evidence of adaptation, **not evidence of better answers**. Cold-start downloads can exceed the 30-minute smoke allocation; pre-stage or request an appropriate walltime if needed.

After that job succeeds, start a **fresh** one-epoch run:

```bash
MODE=full sbatch --account=YOUR_GPU_ACCOUNT --time=02:00:00 training/root_sft/perlmutter.sbatch
```

This uses all 222 training examples and evaluates all 46 validation prompts before and after. The walltime is a requested budget, not a measured runtime estimate. Do not use smoke results as a scientific comparison.

For an existing supported GPU container session, the equivalent entry point is:

```bash
source training/setup.sh
python -m pip install -r training/root_sft/requirements.txt
RUN_DIR=/workspace/artifacts/root-sft/my-fresh-smoke \
  bash training/root_sft/run.sh smoke
```

## 4. Evaluate before versus after

Validation supports development. Keep test questions/results out of training and hyperparameter decisions. Both sides use identical user/system prompts, greedy decoding and a 512-token generation limit. If outputs hit the limit, investigate truncation before scoring; do not silently grade clipped answers as complete.

Give a reviewer `review/paired_review.jsonl` **without** `private_mapping.json`. Each question has randomly ordered A/B answers, a reference and sources. Fill in four integer scores from 0 to 2 for each answer, and a boolean `critical_error`:

| Criterion | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Correctness | Wrong central claim | Mostly right with a substantive error | Scientifically and technically correct |
| Completeness | Does not answer the task | Missing a necessary part | Covers the requested task |
| Assumptions | Invents evidence or ignores a decisive condition | Important caveat unclear | Relevant conditions, units and limits stated |
| Clarity | Unusable or contradictory | Understandable but ambiguous | Clear and directly usable |

Mark `critical_error: true` for an unsafe operation, invented file observation, or central physics/statistics error that invalidates the answer; its total becomes zero. Otherwise the score is the sum divided by 8. Accept valid alternative wording and methods. Use `notes` for disagreements with the key and consult primary documentation. Prefer a second independent reviewer for scientific claims and adjudicate disagreements before reporting a headline score. Do not use exact-string match or keyword overlap as a scientific reward.

Save completed records to a new file, then run:

```bash
python training/root_sft/evaluate.py report \
  --scored /path/to/scored.jsonl \
  --mapping /path/to/review/private_mapping.json \
  --output /path/to/new-score-report.json
```

The report includes paired improvements/regressions, overall means, per-topic/difficulty/category results and an equal-category macro mean. Missing predictions, changed prompts and incomplete scores are rejected. Report sample counts and limitations; these small descriptive results are not a significance test.

Once training settings are frozen, run `inference/run_prompts.py` twice on `test_prompts.jsonl`, once with the resolved baseline snapshot and once with the exported model. Use the same system prompt from the run's `system_prompt.txt`, `--format chat --device cuda --temperature 0 --max-new-tokens 512` for both. Prepare a separate review with `evaluate.py prepare --references .../test_references.jsonl --before ... --after ... --output ...`; score it without further tuning. A snapshot under the old job's `/tmp` may need to be fetched again using the commit recorded in `preflight.json`.

## Verification status

### Run logs

New smoke and full runs automatically stream stdout and stderr to both the terminal
and `RUN_DIR/run.log`, starting after the output directory is created. This includes
preflight/model loading, baseline inference, training, export, weight checks and
post-training inference. Timestamped stage markers identify progress; `training.log`
remains a training-only view. Python output is unbuffered. `status.json` records the
exit code, final stage and elapsed wall-clock time on normal exit or a catchable
failure. A node crash, SIGKILL or scheduler hard kill can prevent that final record.
Commands before `run.sh` (container startup and dependency installation) are outside
this log; batch submission stdout captures them. Early argument/directory errors
also appear only in the calling terminal/job log. No previous run is overwritten.

From another terminal on the host, monitor a new run with:

```bash
tail -f artifacts/root-sft/smoke-interactive-002/run.log
```

The earlier `smoke-interactive-001` predates full-run logging; its existing
`training.log` and saved artifacts remain available, but its complete console
output cannot be reconstructed retroactively.

CPU tests cover deterministic preparation, Parquet/chat round-trips, balancing, grouping, integrity, draft gating, prediction alignment, human scoring, synthetic adapter checks, and success/failure logging. The existing ROOT tests also pass. On September 8, 2026, `smoke-interactive-001` completed two real GPU optimizer steps on one A100 40 GB, exported and reloaded its checkpoint, and produced four before/after answers plus paired review artifacts. All 196 checked adapter layers had nonzero weight deltas. This passes the pipeline smoke gate, not a scientific-quality gate; important answer errors remain. See that run's `inspection.md` for the qualitative review. The newly added whole-run logger has been tested with CPU success/failure cases; it was not present during that GPU run.

## Repository update

`dwkim_dev` was fast-forwarded to `origin/main` at `69e7388` on September 8, 2026, with existing local work reapplied. It now tracks `origin/main`, so `git pull --ff-only` works while the branch can fast-forward. A backup stash named `codex-pre-main-update-20260908` was retained. Do not apply it again to the already-restored files.

After making development commits, fetch and merge `origin/main` deliberately if histories diverge; do not reset away local work. To publish your development branch later use `git push origin dwkim_dev` explicitly. No commit, push or GPU submission was performed by this preparation workflow.
