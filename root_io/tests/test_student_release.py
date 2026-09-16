from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

from students.root_io.build_release import BANK, OUTPUT, build, inline, load_records, render
from students.root_io.worked_exercises import run_lab


class WorkbookParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.answers = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.ids.append(attrs["id"])
        if tag == "details":
            self.answers += 1
            if "open" in attrs:
                raise AssertionError("Answers must initially be hidden")


class StudentReleaseTest(unittest.TestCase):
    def test_schema_and_complete_workbook(self):
        rows = load_records()
        parsed = WorkbookParser()
        parsed.feed(render(rows))
        self.assertEqual(len(parsed.ids), 617)
        self.assertEqual(len(set(parsed.ids)), 617)
        self.assertEqual(parsed.answers, 617)
        for row in rows:
            if row["question_kind"] == "contextual_analysis":
                self.assertTrue(row["context"].startswith("Hypothetical teaching scenario"))

    def test_escaping(self):
        self.assertEqual(inline('`<script>` & "x"'), '<code>&lt;script&gt;</code> &amp; &quot;x&quot;')
        rows = deepcopy(load_records()[:1])
        rows[0]["answer"] = "<img src=x onerror=alert(1)>"
        self.assertNotIn("<img", render(rows))

    def test_missing_record_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "incomplete.jsonl"
            bank.write_text(BANK.read_text().splitlines()[0]+"\n")
            with self.assertRaisesRegex(ValueError, "Expected 617"):
                load_records(bank)

    def test_release_is_current_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = build(out)
            self.assertEqual(manifest["core_concepts"] + manifest["practice_variants"], 617)
            self.assertGreater(manifest["duplicate_answer_groups_crossing_legacy_splits"], 0)
            for filename in ("index.html", "manifest.json"):
                self.assertEqual((out/filename).read_bytes(), (OUTPUT/filename).read_bytes())

    def test_lab(self):
        result = run_lab()
        self.assertEqual(result["selected_event_ids"], [1001, 1002, 1004])
        self.assertAlmostEqual(result["selected_weighted_yield"], 2.9)
        self.assertAlmostEqual(result["selected_weight_variance"], 2.8094)
        self.assertEqual(result["object_histogram_counts"], [2, 3, 2])


if __name__ == "__main__":
    unittest.main()
