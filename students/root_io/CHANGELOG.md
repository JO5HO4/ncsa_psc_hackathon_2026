# Student release — September 2026

The maintained 617-question bank is the source of this release. All IDs are preserved; no questions were deleted. See [editorial_changes.json](editorial_changes.json) for the record-level list of changed fields.

## Content corrections

- Revised 187 answers and 38 questions, including propagated fixes across related scenarios. Updated affected `required_facts` alongside answers so the answer key and its metadata agree.
- Distinguished observable normalization ranges from fit-parameter bounds, required `Save()` when requesting a RooFitResult, and clarified conditional/product PDFs.
- Corrected TH1 drawing/overlay behavior, vector access with TTreeReaderValue, lazy TChain counts, object ownership and legacy-container guidance. Added direct API references to selected corrected entries.
- Clarified weighted yields versus effective sample size, nuisance pulls, systematic correlations, template noise, truth/reco comparisons and overlapping control/signal regions. Removed claims that a systematic mistake always changes uncertainty in one direction.
- Corrected dimensional assumptions about pt versus eta; clarified minimal columns for weighted/unweighted plots, collection-length checks, sorting assumptions and count-only selection versus actual basket I/O.
- Qualified sample-type and weight assumptions: branch names alone do not prove sample provenance, and collision data are not universally unweighted.
- Marked all 333 contextual questions as hypothetical teaching scenarios. They are not observations from real files. Added same-collection assumptions for relevant length-check exercises.
- Normalized eight topic labels to lowercase. The schema now permits dot-separated topic hierarchies, including existing `.b` follow-ups.

## Student materials

- Added an offline HTML workbook with search, topic/difficulty filters, answer reveals and further-reading links. Topic-based grouping labels 182 entries core and 435 practice; it does not claim semantic independence.
- Added six worked exercises on a reproducible-content synthetic ROOT fixture, with numerical checks, and two optional PyROOT API checks. Temporary fixture files are isolated from student data.
- Added schema, completeness, HTML escaping, generated-release freshness and exercise tests. The release manifest records source/workbook digests and duplicate-answer diagnostics.

## Verification and limits

Verified with Python 3.10.18, Uproot 5.6.3, Awkward 2.8.7, NumPy 1.26.4 and jsonschema 4.25.0. The optional checks passed with ROOT 6.38.00. These are the tested versions, not a guarantee for every environment. The fixture contents reproduce, but ROOT file bytes may differ because of timestamps and UUIDs.

All 13 tests passed, as did the six numerical exercises and two optional PyROOT checks. HTML structure and escaping were tested programmatically; an interactive browser/accessibility review has not been performed. External links are retained as further reading; there was no exhaustive link-availability audit.

This was a targeted editorial and executable-example review, not independent expert review of every scientific claim. Instructors should review assigned advanced material before graded use. Machine-training review status remains draft. Repeated answers and related variants still exist across legacy splits: this release is for public learning, not certified SFT/RL training or held-out evaluation.
