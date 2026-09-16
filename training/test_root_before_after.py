import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from subprocess import CompletedProcess

from training.root_before_after import run_stage


class StageTests(unittest.TestCase):
    def test_stage_success_records_status(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch('training.root_before_after.subprocess.run', return_value=CompletedProcess([], 0)):
                run_stage('base', ['python', 'example.py'], {}, out)
            self.assertEqual(json.loads((out / 'base-status.json').read_text())['exit_code'], 0)
            self.assertTrue((out / 'base.log').exists())

    def test_stage_failure_stops_pipeline_and_retains_status(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch('training.root_before_after.subprocess.run', return_value=CompletedProcess([], 17)):
                with self.assertRaisesRegex(RuntimeError, 'failed with exit 17'):
                    run_stage('train', ['bash', 'example.sh'], {}, out)
            self.assertEqual(json.loads((out / 'train-status.json').read_text())['exit_code'], 17)


if __name__ == '__main__':
    unittest.main()
