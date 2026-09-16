import copy
import tempfile
from pathlib import Path
import unittest

from training.root_sft.score_checkpoint_review import aggregate, annotations


class ReviewScoringTest(unittest.TestCase):
    def fixture(self):
        row = dict(id='q1', category='files', scores={
            'A': dict(correctness=2, critical_error=False, notes='Correct.'),
            'B': dict(correctness=1, critical_error=True, notes='Wrong actionable claim.'),
            'C': dict(correctness=1, critical_error=False, notes='Incomplete.'),
            'D': dict(correctness=0, critical_error=False, notes='Does not answer.')})
        return [row], {'q1': dict(zip('ABCD', ['base', 'epoch1', 'epoch2', 'epoch3']))}

    def test_raw_and_critical_gated_are_distinct(self):
        rows, mapping = self.fixture()
        result = aggregate(rows, mapping)
        self.assertEqual(result['models']['epoch1']['overall']['mean_correctness'], 1)
        self.assertEqual(result['models']['epoch1']['overall']['mean_critical_gated'], 0)
        self.assertEqual(result['models']['epoch3']['overall']['critical_errors'], 0)
        self.assertEqual(result['paired']['base_to_epoch1']['regressed'], 1)
        self.assertEqual(result['paired']['base_to_epoch1']['new_critical_errors'], 1)

    def test_reject_incomplete_or_inconsistent_scoring(self):
        rows, mapping = self.fixture()
        for field, value in [('correctness', None), ('correctness', True), ('critical_error', None), ('notes', '')]:
            bad = copy.deepcopy(rows)
            bad[0]['scores']['A'][field] = value
            with self.assertRaises(ValueError):
                aggregate(bad, mapping)
        mapping['q1']['D'] = 'base'
        with self.assertRaises(ValueError):
            aggregate(rows, mapping)

    def test_macro_categories_not_weighted_by_rows(self):
        rows, mapping = self.fixture()
        second = copy.deepcopy(rows[0])
        second['id'] = 'q2'
        third = copy.deepcopy(rows[0])
        third.update(id='q3', category='stats')
        third['scores']['A']['correctness'] = 0
        mapping.update(q2=mapping['q1'].copy(), q3=mapping['q1'].copy())
        result = aggregate(rows + [second, third], mapping)['models']['base']
        self.assertAlmostEqual(result['overall']['mean_correctness'], 4 / 3)
        self.assertEqual(result['macro_category_correctness'], 1)

    def test_duplicate_annotations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.tsv'
            row = '001\t2|Good\t1!|Broken code\t1|Partial\t0|Missing\n'
            path.write_text('id\tA\tB\tC\tD\n' + row)
            self.assertEqual(len(annotations(path)), 1)
            path.write_text(path.read_text() + row)
            with self.assertRaises(ValueError):
                annotations(path)


if __name__ == '__main__':
    unittest.main()
