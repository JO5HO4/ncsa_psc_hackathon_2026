# ROOT-I/O inspection tools

This package provides the bounded, read-only tools used by the `root_io`
dataset family. The model never receives a Python or ROOT command interface.
It can only request structured observations through these functions:

- `list_root_objects`: list keys, ROOT classes, and basic object metadata;
- `describe_tree`: list branch names and types, optionally using glob filters;
- `summarize_branch`: read a bounded event prefix and calculate simple
  statistics for one branch.

Install the small Python dependency set if it is not already available:

```bash
python3 -m pip install -r root_io/requirements.txt
```

All requested file paths must be relative to an evaluator-selected
`allowed_root`. Absolute paths and paths that escape that directory are
rejected. Object, branch, entry, and sample-value counts have hard limits.

## Local example

Generate a tiny, synthetic ROOT file. It is for examples and software tests,
not a replacement for reviewed ATLAS Open Data fixtures.

```bash
python3 root_io/scripts/generate_example_fixture.py
```

Inspect it from the repository root:

```bash
python3 -m root_io.tools \
  --allowed-root data/root_io/generated \
  list-objects diphoton-example.root

python3 -m root_io.tools \
  --allowed-root data/root_io/generated \
  describe-tree diphoton-example.root analysis --pattern 'photon_*'

python3 -m root_io.tools \
  --allowed-root data/root_io/generated \
  summarize-branch diphoton-example.root analysis photon_pt --max-entries 100
```

The CLI writes JSON to stdout and exits with status 2 for a rejected request
or inspection error. Production harness adapters should call the Python
functions and serialize the returned dictionaries without changing their
semantics.

## Tests

```bash
python3 -m unittest discover -s root_io/tests -v
```

Tests generate private temporary fixtures. They do not require downloaded
Open Data.
