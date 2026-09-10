# Run and develop the TRExFitter workflow

Run commands from the repository root. This guide owns setup, execution, and
module layout. The [technical reference](REFERENCE.md) owns verifier criteria,
backend semantics and source provenance.

`runner.py` runs the example analysis configs with the pinned StatAnalysis
container. For configs that read ntuples, it can alternatively use Coffea for
the expensive histogramming (`n`) action and pass the resulting ROOT
histograms to unmodified TRExFitter `w`, `f`, and `s` actions.

The native TREx `.config` is the source of truth for both backends. The Coffea
backend intentionally supports the basic subset exercised by `hyy.config`: `Job`,
`Region`, and `Sample` definitions, flat or jagged branch expressions, nominal
MC weights, and split histogram files. It fails explicitly on `Systematic`
blocks rather than silently producing incomplete histograms.

Inputs are under `data/samples/hyy/{Data,MC}` (16 data and 10 MC files), mounted
as `/workdir/inputs/hyy`. See [data setup](../data/README.md) for the existing
sample layout. `scripts/fetch_hyy_inputs.sh` downloads the Hyy fixture with the
`hf` CLI from `ho22joshua/atlas_opendata`; it requires an existing, empty
`data/samples/hyy` directory and refuses existing `Data`/`MC` paths. Do not run
it over an already prepared sample directory. `runner --check` also checks
that `data/samples/examples` exists. The runner supplies container path and
input mounts; manual environment-variable exports are not required.

## Environment

### Pinned TRExFitter container

The runner uses Podman-HPC with the immutable StatAnalysis 0.8.2 image:

```text
gitlab-registry.cern.ch/atlas/statanalysis@sha256:36a8c06ae90401e3629830c8ffe3b9fcf0b2a1841b28f59e4ecfa49c243051e1
```

The exact digest was inspected on NERSC on 2026-09-10 and contains:

- TRExFitter v1.10.0;
- ROOT v6.40.04;
- Python 3.10.6;
- xRooFit `v0.0.4-22-g5eb77d8`.

The container's `trex-fitter` executable is installed at:

```text
/usr/StatAnalysis/0.8.2/InstallArea/x86_64-el9-gcc14-opt/bin/trex-fitter
```

The image reference is defined in `trex_fitter/runtime.py`. The traditional
backend runs `n`, `w`, `f`, and `s` in this container. The Coffea backend runs
only the replacement `n` implementation in the host-side Python environment;
the unmodified container still runs requested `w`, `f`, and `s` actions.
The corresponding TRExFitter documentation release is the `v1.10.0` tag of
`TRExStats/TRExFitter-Documentation` (commit
`ee86eaa730325cb30534b1032dfe42c917a37930`).

### Host-side Python environment

Host-side dependencies are managed by `uv` using the repository's
`pyproject.toml` and committed `uv.lock`. Load the NERSC Python module first;
`.python-version` selects the tested Python 3.11 series.

The default environment is intentionally small and supports static config
verification:

```bash
module load python
uv sync --locked
uv run --locked python -m trex_fitter.config_verify \
  data/configs/examples/hyy.config
```

Install the optional Coffea histogramming stack on a compute node with:

```bash
uv sync --locked --extra coffea
```

The default `base` schema exposes original branch names exactly as the legacy
flat ntuples and TREx expressions spell them. `atlas-schema` is therefore not
part of the normal workflow. To test its optional collection layout, add
`--extra atlas-schema` to `uv sync` and `uv run`, then pass
`--coffea-schema atlas`. TRExFitter and ROOT remain in the pinned StatAnalysis
container and are not installed by `uv`.

On the tested NERSC Python 3.11 environment, the default verifier environment
occupies about 82 MiB. The complete Coffea plus atlas-schema environment is
about 815 MiB because Coffea brings the broader Scikit-HEP and distributed-I/O
stack. Neither environment includes ROOT, CUDA, or machine-learning frameworks.

## Run

Run histogramming and fitting on an allocated CPU node, not a login node.
For example, on NERSC request `salloc --nodes 1 --qos interactive --time
01:00:00 --constraint cpu --account atlas`, keep the allocation open, and run
the commands below in its node shell. Native `n` can take roughly half an hour
for this fixture; allow headroom. Release the allocation when finished.

Check the container and verify both native analysis validity and Coffea
compatibility, including optional read-only input metadata checks:

```bash
uv run --locked python -m trex_fitter.runner --check
uv run --locked --extra inputs python -m trex_fitter.config_verify \
  data/configs/examples/hyy.config --actions nwsf --check-inputs \
  --json artifacts/trex_fitter/verification.json
```

`--check` starts the container; static verification without `--check-inputs`
does neither container startup nor ROOT-file I/O.

Run only the Coffea replacement for `trex-fitter n`:

```bash
uv run --locked --extra coffea python -m trex_fitter.runner \
  data/configs/examples/hyy.config \
  --backend coffea --actions n --coffea-workers 8 \
  --coffea-stage-dir /tmp/trex-hyy-${SLURM_JOB_ID:?Run on an allocated node} \
  --output-dir artifacts/trex_fitter/coffea-run/output \
  --log-dir artifacts/trex_fitter/coffea-run/logs
```

This writes `output/hyy/Histograms/hyy_<region>_histos.root`, including the `_orig`,
nominal, and `_regBin` objects that the later TREx actions expect. Running all
stages uses Coffea only for `n`:

```bash
uv run --locked --extra coffea python -m trex_fitter.runner \
  data/configs/examples/hyy.config --backend coffea --actions nwsf \
  --coffea-workers 8 \
  --coffea-stage-dir /tmp/trex-hyy-${SLURM_JOB_ID:?Run on an allocated node} \
  --output-dir artifacts/trex_fitter/coffea-run/output \
  --log-dir artifacts/trex_fitter/coffea-run/logs
```

Use `--coffea-maxchunks 1` for a quick smoke test. Use `--output-dir` to keep a
benchmark or complete run under ignored `artifacts/`. When later actions are
requested, the runner starts TRExFitter in that output directory so it reads
the Coffea-produced Job directory in place.

Run the native reference chain into a separate directory:

```bash
uv run --locked python -m trex_fitter.runner \
  data/configs/examples/hyy.config --backend trex --actions nwsf \
  --output-dir artifacts/trex_fitter/native-run/output \
  --log-dir artifacts/trex_fitter/native-run/logs
```

Choose a fresh artifact directory for each independent run. Omitting
`--output-dir` uses the repository root, so the examples always provide it.
Both `--actions nwsf` and `--actions n w s f` work; native TRExFitter chooses
the internal execution order. For histogramming alone use `--actions n`.

Related native stages are passed as one TRExFitter action string (for example,
`wfs`) and run in one container. The pinned image is cached locally by
Podman-HPC; `--rm` removes the finished container, not the 8 GiB image. Runner
output is streamed immediately while separate stdout/stderr logs are retained.

`--coffea-stage-dir` is strongly recommended on NERSC. It performs one
sequential copy of each configured ROOT file to node-local storage before
Coffea starts its parallel, branch-oriented reads. Staging time is included in
the reported wall time, and complete files already present in the stage
directory are reused by file size. See the [staging caveat](REFERENCE.md#supported-boundary)
before using non-Hyy layouts or reusing a stage after input changes.

Compare a completed candidate against a TRExFitter reference:

```bash
uv run --locked --extra coffea python -m trex_fitter.coffea_backend.compare \
  artifacts/trex_fitter/native-run/output/hyy \
  artifacts/trex_fitter/coffea-run/output/hyy \
  --json artifacts/trex_fitter/coffea-run/comparison.json
```

Compatibility means identical ROOT paths, bin edges, ROOT histogram/axis
titles, and numerically consistent bin contents and variances.

## Fast analysis config verification

See [the verifier reference](REFERENCE.md#config-verification) for default checks, optional
`--check-inputs`, report fields, and differences from native `ReadFullConfig`.

Validate both the complete basic analysis structure used by `hyy.config`—Job,
Fit, Region, Sample, and NormFactor blocks plus their cross-references—and
Coffea `n` compatibility without opening ROOT files or starting TRExFitter:

```bash
uv run --locked python -m trex_fitter.config_verify \
  data/configs/examples/hyy.config
```

The single Pydantic-backed verifier returns one overall `valid` value plus
granular `analysis_valid` and `coffea_compatible` values in its Python/JSON
report. It reports errors rather than warnings for unsupported operations. It
implements the project's source-grounded basic v1.10 profile, not every
advanced TRExFitter option and not the physics intent of an analysis.

Run the automated tests with pytest's structured, colored terminal output:

```bash
uv run --locked --extra coffea pytest -v --color=yes
```

## JSON reporting and timeouts

Use the same runner for execution and an optional machine-readable report:

```bash
uv run --locked --extra coffea python -m trex_fitter.runner \
  data/configs/examples/hyy.config --backend coffea --actions nwsf \
  --coffea-workers 8 --output-dir artifacts/trex_fitter/run/output \
  --log-dir artifacts/trex_fitter/run/logs \
  --json artifacts/trex_fitter/run/result.json --timeout 3600
```

Use `--json -` for JSON-only stdout; live output goes to stderr in that mode.
Otherwise logs stream normally. The report includes success, child exit code,
elapsed seconds, timeout status, backend/actions, output/log directories,
errors, container-cleanup errors, and observed significance when found in this
invocation's stdout. Missing or nonfinite significance is `null`; this is an
execution report, not a physics-quality assessment or a config-verifier report.

`--timeout SECONDS` is optional (no limit by default) and covers the complete
chain, including Python startup, Coffea processing/workers, and native stages.
Supervision terminates the owned process group and forcibly removes containers
identified by this invocation's CID files. Cleanup has a bounded grace period,
so wall time can exceed the requested execution limit. A timeout exits 124;
execution failures exit 1, while top-level argument errors use argparse's exit 2.
Partial outputs are kept for diagnosis, not promoted to valid results.

Reporting or timeout supervision creates a unique `run-*` directory below
`--log-dir`, retaining `runner.stdout.log`, `runner.stderr.log`, native logs,
and container IDs. It never scans old logs for significance. The supervisor
invokes this same runner without its reporting/timeout options; there is no
second execution CLI. It also supports `--check` and `--dry-run`; dry-run
reports describe a plan, not a completed analysis.

`evaluate.py`, the public `mock.py`/`--mock` mode, and the old runtime CLI
were removed. Fake subprocesses live only in the tests. Static verification
remains a separate command; native parsing is authoritative for native runs.

## Module ownership

| Module | Responsibility |
| --- | --- |
| `config_format.py` | Shared config syntax/errors, without backend dependencies |
| `config_verify.py`, `schema.py`, `semantics.py`, `input_checks.py` | Unified report, native schema, static semantics, optional input validation |
| `runner.py` | Primary native/Coffea orchestration |
| `runtime.py` | Reusable container, subprocess, timeout, logging, and result-parsing helpers; no CLI |
| `coffea_backend/` | Histogram config, expressions, processing, writing, comparison, capability checking |
| `scripts/fetch_hyy_inputs.sh` | Still-used Hyy input download helper |
| `schemas/`, `../tests/trex_fitter/` | Pinned native schemas and regression tests |

Use `python -m trex_fitter.runner` for all analysis execution and
`python -m trex_fitter.config_verify` for standalone verification.
Import syntax helpers directly from `config_format`. `coffea_backend.verify`
remains the internal capability checker; it does not replace the unified
analysis verifier. The older runtime CLI and its duplicate sequence/parallel-region
orchestration were retired.

The redundant `scripts/trex.py` and unused structural-only `verifier.py` were
retired.
Backward compatibility with those command paths is not maintained. Training
and inference keep their separate model environments and modules.

See the reference for the [operation map](REFERENCE.md#backend-operation-map).
Reports, ROOT files,
logs, and smoke-test output belong under ignored `artifacts/`, not source folders.
