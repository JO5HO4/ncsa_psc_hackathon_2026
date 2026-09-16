from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from data.root_io.validate_tasks import validate_record


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads(
    (REPO_ROOT / "data/root_io/schema/task.schema.json").read_text(encoding="utf-8")
)
EXAMPLE = json.loads(
    (REPO_ROOT / "data/root_io/examples/inspect-diphoton.example.json").read_text(
        encoding="utf-8"
    )
)


class DatasetRecordTest(unittest.TestCase):
    def test_example_is_valid(self) -> None:
        validate_record(EXAMPLE, SCHEMA)

    def test_rejects_tool_outside_allowlist(self) -> None:
        record = deepcopy(EXAMPLE)
        record["allowed_tools"] = ["describe_tree"]
        with self.assertRaisesRegex(ValueError, "not allowed"):
            validate_record(record, SCHEMA)

    def test_reviewed_record_requires_real_checksum(self) -> None:
        record = deepcopy(EXAMPLE)
        record["metadata"]["review_status"] = "reviewed"
        with self.assertRaisesRegex(ValueError, "placeholder SHA-256"):
            validate_record(record, SCHEMA)


if __name__ == "__main__":
    unittest.main()
