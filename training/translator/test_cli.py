import json
import tempfile
import unittest
from pathlib import Path

from training.translator import cli


class CliRunTest(unittest.TestCase):
    def test_simple_qa_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "questions.jsonl"
            rows = [
                {"id": "q1", "question": "Print 1.", "answer": "root -l -b -q -e 'x1'", "category": "mathematics"},
                {"id": "q2", "question": "Print 2.", "answer": "root -l -b -q -e 'x2'", "category": "mathematics"},
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output_dir = tmp_path / "out"

            manifest = cli.run(
                source=source, source_format="simple-qa", task_name="root_command",
                model_name="qwen3.5", split="train", output_dir=output_dir,
                repo_root=tmp_path, expected_count=2,
            )

            self.assertEqual(manifest["records"], 2)
            self.assertTrue((output_dir / "train.jsonl").is_file())
            self.assertTrue((output_dir / "train.parquet").is_file())
            self.assertTrue((output_dir / "train.manifest.json").is_file())
            written = [json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines()]
            self.assertEqual([row["id"] for row in written], ["q1", "q2"])
            self.assertEqual(written[0]["tools"], "[]")
            self.assertIs(written[0]["enable_thinking"], False)

    def test_wrong_expected_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "questions.jsonl"
            source.write_text(json.dumps({"id": "q1", "question": "Q", "answer": "root -l -b -q -e 'x'"}) + "\n")
            with self.assertRaisesRegex(ValueError, "Expected 5 rows"):
                cli.run(
                    source=source, source_format="simple-qa", task_name="root_command",
                    model_name="qwen3.5", split="train", output_dir=tmp_path / "out",
                    repo_root=tmp_path, expected_count=5,
                )

    def test_invalid_answer_stops_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "questions.jsonl"
            source.write_text(json.dumps({"id": "q1", "question": "Q", "answer": "not a root command"}) + "\n")
            output_dir = tmp_path / "out"
            with self.assertRaises(ValueError):
                cli.run(
                    source=source, source_format="simple-qa", task_name="root_command",
                    model_name="qwen3.5", split="train", output_dir=output_dir,
                    repo_root=tmp_path,
                )
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
