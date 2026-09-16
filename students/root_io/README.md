# ROOT and HEP: student self-study edition

Open [index.html](index.html) in a browser. It contains all **617 questions and answers**, works offline, and needs no Python, account, server, or ROOT installation. Search by topic or question ID, filter by difficulty, and reveal answers after attempting each question. External documentation links require internet.

The topic-based learning path contains **182 core entries and 435 practice variants**. These counts are not a claim of 617 orthogonal skills. Core means the first question for a topic; `.b` follow-ups and repeated scenarios are practice. An instructor may assign other questions first.

## Suggested learning path

1. Start with introductory file, object, tree and branch questions. Learn to separate observed metadata from assumptions about physics.
2. Study jagged collections, safe indexing, selection and weights, then complete the lab below.
3. Continue to histograms, uncertainties, plotting and fitting. Explain the statistical assumptions, not just the method names.
4. Attempt advanced RooFit/RooStats and HEP interpretation questions after learning likelihoods, covariance and nuisance parameters. Use the sources to explore beyond these short answers.

For each answer, check: Is the key concept correct? Are assumptions and units stated? Could someone apply the advice safely? For code exercises, include your code and actual output. A matching final number without sound reasoning is not sufficient.

## Hands-on lab

Run from the repository root with Python 3.10 or newer. A virtual environment is recommended; the standard lab needs Uproot, Awkward, NumPy and jsonschema, **not PyROOT**:

```bash
python -m venv .venv-root-students
source .venv-root-students/bin/activate
python -m pip install -r students/root_io/requirements-tested.txt
python students/root_io/worked_exercises.py
```

The worked solution creates a four-event synthetic ROOT file in a fresh temporary directory, checks its answers and removes the temporary file on exit. It does not open or modify your physics data. To attempt the exercises yourself, use `create_example_fixture` from `root_io.fixtures` inside your own `tempfile.TemporaryDirectory` and inspect it with Uproot. See [worked_exercises.py](worked_exercises.py) only after trying the tasks.

Teaching conventions: `photon_pt` is in GeV, `photon_eta` is dimensionless, the photon arrays are aligned, and this fixture's photons are ordered by decreasing pt. The weight is an illustrative per-event number, not a luminosity-normalized physics prediction. These declarations apply to this lab only; branch names do not establish units or sorting in other files. The `cutflow` histogram is a **separate synthetic example**, not a cutflow computed from the four-event tree.

| Exercise | Task | Reference check |
| --- | --- | --- |
| 1. Inventory | List stored object classes and the analysis entry count. | `analysis`: TTree, 4 entries; `cutflow`: TH1D |
| 2. Schema | Check the photon count against both jagged-array lengths in every event. | All four events agree |
| 3. Selection | Require at least two photons and `trigP`, then read the second photon's pt safely. | Event IDs 1001, 1002, 1004; pt 44, 38, 42 GeV |
| 4. Weights | Compute the selected weighted yield and its variance estimate for independent events using sum(w²). | Yield 2.90; variance 2.8094 |
| 5. Histogram | Histogram all photons in selected events with edges [0, 40, 60, 80] GeV, first unweighted then using broadcast event weights. | Counts [2, 3, 2]; weighted contents [1.88, 2.90, 1.93] |
| 6. Cutflow | Interpret [100, 80, 42] as cumulative counts and calculate successive efficiencies. | 0.80 and 0.525; summing the bins does not count unique events |

Exercise 5 uses NumPy binning, whose final bin includes its right edge; ROOT TH1 treats the upper axis edge as overflow. No fixture value lies on that edge. Objects from the same event are correlated: broadcasting weights produces object-level contents, but does not make the objects independent statistical events. Exercise 4's event-yield variance must not be blindly reused for arbitrary derived observables.

Extension: change a copy of the fixture generator to include an empty photon list, a negative event weight, or a count/array mismatch. Predict which validation or result should change before rerunning. Do not alter real input files to test this.

If your environment already provides PyROOT, these additional checked examples demonstrate histogram ownership and reading a vector through TTreeReaderValue:

```bash
python students/root_io/worked_exercises.py --with-pyroot
```

Do not install an unrelated package named `ROOT` to satisfy this optional step; use your course's supported ROOT environment. ROOT's [ownership manual](https://root.cern/manual/object_ownership/) and [TTreeReader examples](https://root.cern/doc/master/classTTreeReader.html) explain these APIs. API details can vary by ROOT release.

## Instructor distribution and maintenance

Distribute `index.html` for a standalone workbook, or copy this directory to a static course site. Include this README for the lab guide. The executable lab needs this repository's `root_io` package, so distribute the repository as well if students will run it. No service or backend is required. Answers are hidden for self-study, **not protected from inspection**; this is not an exam system.

After editing the maintained bank, rebuild and test from the repository root:

```bash
python students/root_io/build_release.py
python -m unittest discover -s root_io/tests -v
python students/root_io/worked_exercises.py
```

The builder validates all records and writes the workbook plus [manifest.json](manifest.json), which records counts and SHA-256 digests. It never regenerates or replaces the source bank. Older 50/500-question generators are historical drafts, not the source of this release.

## Review scope and training limitations

This edition incorporates targeted technical/editorial corrections, schema validation and executed synthetic exercises; it is **not independently expert-certified**. See [CHANGELOG.md](CHANGELOG.md). Instructors should review their assigned advanced-statistics questions against the course's conventions before using them for graded assessment. Report corrections with the question ID, proposed answer, relevant documentation and software version.

Hypothetical contexts are explicitly labeled. References provide further reading; not every link is a sentence-level citation, and none validates invented scenario contents. The short answers supplement, rather than replace, ROOT documentation or an experiment's analysis recommendations.

All machine-training `review_status` fields remain `draft`, and `rl_eligible` remains `false`. Legacy train/validation/test labels are preserved but are **not leakage-free**: repeated answers, related topics and scenario variants can cross splits. Publishing the answer key also makes these unsuitable as secret evaluation questions. Before SFT, perform independent content review and concept/scenario-aware deduplication and splitting. Before RL, add executable tasks with separate fixtures, verified outcomes and a tested reward function. This student release does not certify either training pipeline.
