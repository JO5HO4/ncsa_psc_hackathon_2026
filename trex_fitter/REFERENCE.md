# TRExFitter technical reference

For setup and executable commands, start with the [run guide](README.md).
This page owns verifier rules, backend semantics and source provenance. Generated validation reports belong
in ignored `artifacts/`.

## Config verification

One command always checks native schema validity, supported static semantics,
and Coffea compatibility. It opens no ROOT files by default:

```bash
uv run --locked python -m trex_fitter.config_verify \
  data/configs/examples/hyy.config --actions nwsf
```

`--actions n w s f` and `--actions nwsf` are equivalent. The default is `nwfs`.
Use `--actions n` when only histogramming is planned. Native TRExFitter executes
combined actions in its own internal order, regardless of letter order.

### Default criteria

- Every supplied setting is checked against the pinned v1.10.0 native schema.
- Basic Job/Fit/Region/Sample/NormFactor values and ranges are checked.
- Fit actions (`f`, `l`, `s`, `r`, `i`, `x`) require a non-validation region.
- `POIAsimov` must be a finite scalar for one POI, or declared `name@value` pairs.
- POIs can resolve to norm-factor blocks, inline sample norm factors, nuisance
  parameters (including `alpha_` names), template parameters, shape-expression
  parameters, and EFT parameters from `EFTValue` for nonsplit EFT fits.
- Sample, norm-factor, shape-factor, and systematic references are checked;
  `all` and `none` are recognized. Common systematic drop/keep references are
  also checked.
- Norm-factor attachments must be unique per sample/region. Repeated names in
  disjoint attachments are allowed.
- `NumCPU`, `ToysHistoNbins`, `RankingNPfraction`, and configured scan step
  counts are checked. Ranking lists must match the number of POIs.
- Coffea selection/weight syntax and supported histogram operations are checked
  independently. A valid native setting need not be supported by Coffea.

Rules have stable diagnostic codes. The new semantic diagnostics include
source references into `Root/ConfigReader.cc` at pinned commit
`52e62c30a1faf1ca9fdfe0db74160bb9e00e86e9`.

Scan step ranges are deliberately stricter than native behavior: TRExFitter
warns and replaces out-of-range values with defaults; this verifier reports
an error so generated configs must state a usable value explicitly.

### Optional input checks

```bash
uv run --locked --extra inputs python -m trex_fitter.config_verify \
  data/configs/examples/hyy.config --check-inputs --actions nwsf \
  --json artifacts/trex_fitter/verification.json
```

The `inputs` extra supplies Uproot; the existing `coffea` extra also supplies it.
These checks inspect metadata and histogram axes, without reading event arrays:

- For the supported Coffea NTUP subset, resolve every configured file pattern,
  check each ROOT file and tree, and check branches used by selections,
  weights, and region variables.
- For simple nominal HIST configs, check files, named TH1 objects, and matching
  edges across samples within a region. One unambiguous HistoPath/HistoFile/
  HistoName per sample-region is required.
- Unsupported input layouts receive an `input_coverage` error. Friend trees,
  advanced suffix combinations, systematic variations, and generated inputs
  are not claimed as checked.

`--project-dir` specifies the host root corresponding to `/workdir`. Its
`data/samples` maps to `/workdir/inputs`; the repository root is the default.
Input checks are read-only but their time depends on the number of files and
filesystem latency. No file sampling is performed.

The JSON/Python report contains `analysis_valid`, `coffea_compatible`, and
`inputs_valid` (`null` when not requested), plus separate diagnostics and
`input_files_checked`. Overall `valid` requires both default checks and, when
requested, passing input checks. Thus a native HIST analysis can have
`analysis_valid=true`, `inputs_valid=true`, but `coffea_compatible=false`.

### Differences from ReadFullConfig

Native `ReadFullConfig` builds an effective analysis, applying defaults,
inheritance, name normalization, command-line overrides, generated parameters,
and template expansions. This verifier does not build that complete model.
It does not yet reproduce full MultiFit, unfolding, EFT splitting, morphing,
shape-factor bin consistency, all sample arithmetic/dependency checks, or every
conditional rule in `PostConfig`. Some inherited input configurations remain
outside the basic static profile.

Metadata checks do not validate ROOT-only formulas, runtime array shapes,
selection yields, numerical fit stability, or physics intent. Native parsing
and actual execution remain necessary for those cases. No native-parser or
event-sampling mode is included here.

## Source provenance

The source reference is TRExFitter v1.10.0, commit
`52e62c30a1faf1ca9fdfe0db74160bb9e00e86e9`. The matching documentation release
is tag `v1.10.0`, commit `ee86eaa730325cb30534b1032dfe42c917a37930`, in
`TRExStats/TRExFitter-Documentation`. Local source and documentation clones
live outside this repository; they are not dependencies installed by uv.

The bundled [schema snapshots](schemas/v1.10.0/) preserve `jobSchema.config`
and `multiFitSchema.config` from that source revision, apart from trailing
blank lines. Validation follows `Root/ConfigParser.cc`: slash-separated
alternatives and comma-separated parameter types. Numeric tokens must also
be complete and finite; Python validation does not accept the numeric-prefix
parsing of `std::stoi`/`std::stod`.

Schema checks cover allowed settings and declared types across blocks, not
all native semantics. The [pinned container](README.md#pinned-trexfitter-container)
remains authoritative for execution.

## Backend operation map

The Coffea backend implements the nominal NTUP operations needed by
`data/configs/examples/hyy.config`, not the entire TRExFitter language.
Paths below are relative to `trex_fitter/` or the pinned native source tree.

| Atomic operation | Backend implementation | Native source | Verification |
| --- | --- | --- | --- |
| Parse Job, Region, Sample | `config_format.py`, `coffea_backend/config.py` | `Root/ConfigReader.cc` | Parser tests and Hyy load |
| Verify supported nominal subset | `coffea_backend/verify.py` | Native configuration semantics | Pydantic and unsupported-field tests |
| Resolve paths/files/tree | `coffea_backend/config.py` | `Root/TRExFit.cc:FullNtuplePaths` | Input inventory and full run |
| Evaluate scalar/jagged expressions | `coffea_backend/expressions.py` | `Root/Common.cc`, `TTree::Draw`, lines 253–269 | ROOT oracle cases and histogram comparison |
| Apply sample and region selections | `coffea_backend/processor.py` | `Root/TRExFit.cc:FullSelection` | Full Hyy comparison |
| MC weights/luminosity, unit data weight | `coffea_backend/processor.py` | `Root/TRExFit.cc:FullWeight`, `Root/NtupleReader.cc` | Full Hyy comparison |
| Weighted bins with sumw2 | `coffea_backend/processor.py` | `Root/NtupleReader.cc:GetHistogram`, `Root/Common.cc` | Contents and variances |
| Fold underflow/overflow | `coffea_backend/writer.py` | `Root/Common.cc:MergeUnderOverFlow` | Flow unit test |
| Repair nonpositive background bins | `coffea_backend/writer.py` | `Root/SampleHist.cc:FixEmptyBins`, around line 640 | Edge-case tests and full comparison |
| Write ROOT hierarchy and metadata | `coffea_backend/writer.py` | `Root/SampleHist.cc`, around lines 496–512 | ROOT round trip and 126-object comparison |
| Compare against native output | `coffea_backend/compare.py` | Observable output contract | Paths, classes, axes, titles, contents, variances |

### Missing indices and Boolean guards

Selections follow `TTree::Draw`, not ordinary short-circuit semantics. A missing
fixed collection index excludes that event even inside a Boolean guard such
as `jet_n<2 || jet_pt[1]>25` or `!(jet_n>=2 && jet_pt[1]>25)`. This behavior was
measured in the pinned ROOT 6.40.04 container. Preserve missing operands through
AND/OR/NOT and reject missing final selections. Numeric operands still use
logical truth values (nonzero is true), not bitwise integer operations.

Replacing missing operands with known truth values admitted extra zero-/one-jet
events into all five Hyy jet-veto categories. It caused 105 of 126 histogram
objects to differ; the 2-jet category's 21 objects remained unchanged. The
regression tests in `tests/trex_fitter/test_review_fixes.py` record the native
oracle expectations. A config verifier cannot determine the intended physics
selection from syntax or branch metadata alone.

Backend equivalence and physics intent are separate requirements. If the
analysis should retain low-jet events, compute an explicitly safe scalar flag
such as `passes_2jet` in preprocessing and select `!passes_2jet`. Validate that
changed analysis with native execution and establish a new baseline; do not
silently change only the fast backend or assume another OR guard fixes it.

### Supported boundary

Only fixed-width, one-dimensional, nominal NTUP histogramming with the input
layout and selection/weight structure used by Hyy is supported. Fit and
NormFactor operations are handled downstream by unmodified TRExFitter.
Systematics, alternate input precedence, Job/Region weights, aliases, friend
trees, variable-width bins, sample arithmetic, smoothing, and unfolding are
rejected by the fast backend. Native acceptance of a setting does not imply
Coffea support. “Accepted” is not a claim of full native equivalence.

Input staging currently keys cached files by sample and basename and reuses
equal-size files. Use a fresh per-job staging directory and ensure basenames
within each sample are unique; this is not a content-verified cache. The Hyy
validation inventory has no such collisions. Do not assume arbitrary input
layouts are safe just because static verification passes.
