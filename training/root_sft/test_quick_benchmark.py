import json
from collections import Counter
from pathlib import Path
import unittest

from training.root_sft.quick_benchmark import REPO, BANK, select, topic, normalized


class QuickBenchmarkTest(unittest.TestCase):
    def test_balanced_reproducible_unique_and_no_sealed_groups(self):
        bank = [json.loads(x) for x in BANK.read_text().splitlines()]
        assignments = [json.loads(x) for x in (REPO/'data/root_io/splits/root-sft-v1/assignments.jsonl').read_text().splitlines()]
        rows = select(bank, assignments)
        self.assertEqual(rows, select(bank, assignments))
        self.assertEqual(len(rows), 50)
        self.assertEqual(dict(Counter(r['category'] for r in rows)),
                         dict(files_io=10, trees_analysis=10, hep_interpretation=10, histograms_plotting=10, statistics=10))
        self.assertEqual(len({topic(r) for r in rows}), 50)
        self.assertEqual(len({normalized(r['answer']) for r in rows}), 50)
        sealed_groups = {r['group'] for r in assignments if r['split'] == 'test'}
        sealed_ids = {r['id'] for r in assignments if r['group'] in sealed_groups}
        self.assertFalse(sealed_ids & {r['id'] for r in rows})


if __name__ == '__main__':
    unittest.main()
