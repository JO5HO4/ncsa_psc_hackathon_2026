import json
import tempfile
import unittest
from pathlib import Path

from training.translator.build_parquet import sha256, write_jsonl, write_parquet


class BuildParquetTest(unittest.TestCase):
    def test_write_jsonl_and_parquet_roundtrip(self):
        rows = [{"id": "a", "messages": [{"role": "user", "content": "hi", "tool_calls": None, "tool_call_id": None}]}]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            jsonl_path = directory / "split.jsonl"
            parquet_path = directory / "split.parquet"
            write_jsonl(rows, jsonl_path)
            write_parquet(rows, parquet_path)
            self.assertEqual(json.loads(jsonl_path.read_text().splitlines()[0]), rows[0])
            self.assertTrue(parquet_path.is_file())
            self.assertEqual(len(sha256(jsonl_path)), 64)

    def test_write_jsonl_refuses_overwrite(self):
        rows = [{"id": "a"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.jsonl"
            write_jsonl(rows, path)
            with self.assertRaises(FileExistsError):
                write_jsonl(rows, path)

    def test_write_parquet_refuses_overwrite(self):
        rows = [{"id": "a"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.parquet"
            write_parquet(rows, path)
            with self.assertRaises(FileExistsError):
                write_parquet(rows, path)


if __name__ == "__main__":
    unittest.main()
