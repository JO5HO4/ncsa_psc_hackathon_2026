# TRExFitter

This directory contains the pinned upstream TRExFitter source and the small
local tools used to preflight and run a configuration.

```text
trex_fitter/
├── source/       # recursive upstream TRExFitter submodule, pinned to v1.10.0
├── runner.py     # source-build launcher
├── verifier.py   # inexpensive static config preflight
└── README.md
```

Clone the repository recursively so that TRExFitter and its nested
dependencies are present:

```bash
git clone --recurse-submodules git@github.com:JO5HO4/ncsa_psc_hackathon_2026.git
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

Build the pinned source in an ATLAS environment:

```bash
source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
asetup StatAnalysis,0.5.1
cmake -S trex_fitter/source -B trex_fitter/source/build
cmake --build trex_fitter/source/build -j 4
```

Preflight a configuration before scheduling an expensive run:

```bash
python3 trex_fitter/verifier.py data/trex_config/FitExample.config
```

Run selected TRExFitter actions after a clean preflight:

```bash
python3 trex_fitter/runner.py data/trex_config/FitExample.config --actions n w f s
```

By default, runner logs are written outside this directory to
`artifacts/trex_fitter/`. Use `--log-dir` to choose another location.

`verifier.py` catches only inexpensive structural problems. A successful
preflight does not guarantee that workspace construction or fitting will
succeed; those are the responsibility of the pinned upstream executable.
