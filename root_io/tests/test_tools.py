from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from root_io.fixtures import create_example_fixture
from root_io.tools import (
    RootInspectionError,
    describe_tree,
    list_root_objects,
    summarize_branch,
)


class RootToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = create_example_fixture(self.root / "fixture.root")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_list_objects(self) -> None:
        result = list_root_objects("fixture.root", allowed_root=self.root)
        by_name = {item["name"]: item for item in result["objects"]}
        self.assertEqual(by_name["analysis"]["entries"], 4)
        self.assertEqual(by_name["cutflow"]["dimensions"], 1)

    def test_describe_tree_filters_branches(self) -> None:
        result = describe_tree(
            "fixture.root",
            "analysis",
            allowed_root=self.root,
            patterns=["photon_*"],
        )
        self.assertEqual(result["entries"], 4)
        self.assertEqual(
            {item["name"] for item in result["branches"]},
            {"photon_n", "photon_pt", "photon_eta"},
        )

    def test_summarize_jagged_numeric_branch(self) -> None:
        result = summarize_branch(
            "fixture.root",
            "analysis",
            "photon_pt",
            allowed_root=self.root,
        )
        self.assertEqual(result["entries_read"], 4)
        self.assertEqual(result["value_count"], 8)
        self.assertEqual(result["minimum"], 19.0)
        self.assertEqual(result["maximum"], 71.0)

    def test_rejects_escape_and_absolute_paths(self) -> None:
        with self.assertRaises(RootInspectionError):
            list_root_objects("../fixture.root", allowed_root=self.root)
        with self.assertRaises(RootInspectionError):
            list_root_objects(self.fixture, allowed_root=self.root)

    def test_limits_are_enforced(self) -> None:
        with self.assertRaises(RootInspectionError):
            summarize_branch(
                "fixture.root",
                "analysis",
                "photon_n",
                allowed_root=self.root,
                max_entries=10_001,
            )


if __name__ == "__main__":
    unittest.main()
