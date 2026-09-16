import json
from pathlib import Path
import tempfile
import unittest

from training.root_sft.file_tasks import build, digest, evaluate, execute, write_jsonl


class FileTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='root-filetasks-test-')
        cls.root = Path(cls.temp.name)
        cls.release = cls.root/'release'
        build(cls.release)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def predictions(self, path):
        refs = [json.loads(x) for x in (self.release/'validation/references.jsonl').read_text().splitlines()]
        rows = [dict(id=r['id'], prompt=r['prompt'], record_type='completion', output=json.dumps(r['reference_plan'])) for r in refs]
        write_jsonl(path, rows)
        return rows

    def test_balance_and_disjoint_fixtures(self):
        manifest = json.loads((self.release/'manifest.json').read_text())
        self.assertEqual(manifest['reference_checks_passed'], 60)
        groups = []
        for split, count in [('train', 6), ('validation', 2), ('test', 2)]:
            self.assertEqual(set(manifest['counts'][split].values()), {count})
            rows = [json.loads(x) for x in (self.release/split/'references.jsonl').read_text().splitlines()]
            groups.append({r['split_group'] for r in rows})
            for r in rows:
                self.assertNotIn('expected', json.loads((self.release/split/'prompts.jsonl').read_text().splitlines()[0]))
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_oracle_roundtrip(self):
        path = self.root/'oracle.jsonl'
        self.predictions(path)
        result = evaluate(self.release, 'validation', path, self.root/'oracle-score')
        self.assertEqual(result['passed'], 12)

    def test_wrong_plan_and_malicious_code_fail(self):
        path = self.root/'wrong.jsonl'
        rows = self.predictions(path)
        rows[0]['output'] = '__import__("os").system("echo unsafe")'
        plan = json.loads(rows[2]['output'])
        plan['threshold'] = -999
        rows[2]['output'] = json.dumps(plan)
        write_jsonl(path, rows)
        result = evaluate(self.release, 'validation', path, self.root/'wrong-score')
        self.assertEqual(result['passed'], 10)

    def test_no_arbitrary_paths_or_overwrite(self):
        fixture = self.release/'validation/fixture_7.root'
        before = digest(fixture)
        with self.assertRaises(ValueError):
            execute(dict(operation='class', object='Records_7', file='/etc/passwd'), fixture, self.root)
        self.assertEqual(digest(fixture), before)
        with self.assertRaises(FileExistsError):
            build(self.release)

    def test_missing_predictions_rejected(self):
        path = self.root/'missing.jsonl'
        rows = self.predictions(path)
        write_jsonl(path, rows[:-1])
        with self.assertRaises(ValueError):
            evaluate(self.release, 'validation', path, self.root/'missing-score')


if __name__ == '__main__':
    unittest.main()
