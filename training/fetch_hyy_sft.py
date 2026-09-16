#!/usr/bin/env python3
"""Download the pinned Hyy tool-use SFT Parquet release with SHA-256 checks."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen


DATASET = "cxyang-ucb/hyy-sft"
REVISION = "e479270a79f46f1a13d00d6af2cf0433fcb49ab5"
FILES = {
    "train.parquet": "53bb9baa031775a3714bc48576a08283ee105ca6db66fead45cc167d0c843f16",
    "validation.parquet": "5e5a8fe3c788342d2a70e3d8f66518e7ee2e7ec65449e6f0ba0737c0d42b616c",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name, expected, output_dir, overwrite):
    output = output_dir / name
    if output.exists() and sha256(output) == expected:
        print("Verified existing {}".format(output))
        return
    if output.exists() and not overwrite:
        raise ValueError("{} has an unexpected SHA-256; use --overwrite to replace it".format(output))

    url = "https://huggingface.co/datasets/{}/resolve/{}/data/parquet/{}".format(DATASET, REVISION, name)
    with urlopen(url) as response, tempfile.NamedTemporaryFile(dir=str(output_dir), delete=False) as temporary:
        shutil.copyfileobj(response, temporary)
        temporary_path = Path(temporary.name)
    actual = sha256(temporary_path)
    if actual != expected:
        temporary_path.unlink()
        raise ValueError("{} SHA-256 mismatch: expected {}, got {}".format(name, expected, actual))
    os.replace(str(temporary_path), str(output))
    print("Downloaded and verified {}".format(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        fetch(name, expected, args.output_dir, args.overwrite)
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "files": FILES,
    }
    (args.output_dir / "source.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
