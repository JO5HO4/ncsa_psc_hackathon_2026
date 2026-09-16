"""Offline tests: trusted ROOT reference execution and checkpoint review integrity."""
import json
from pathlib import Path
import tempfile
import unittest

from training.root_sft.compare_checkpoints import assemble
from training.root_sft.verified_examples import build, CASES, fixture


class VerifiedExamplesTest(unittest.TestCase):
    def test_balanced_executed_pack_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'pack'
            rows = build(output)
            report = json.loads((output / 'verification.json').read_text())
            self.assertEqual(len(rows), 12)
            self.assertEqual(set(report['counts'].values()), {2})
            self.assertEqual(len(report['counts']), 6)
            self.assertEqual(len({r['split_group'] for r in rows}), 1)
            self.assertEqual(len({r['question'] for r in rows}), 12)
            with self.assertRaises(FileExistsError):
                build(output)

    def test_published_code_runs_and_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = build(root / 'pack')
            path = root / 'input.root'
            fixture(path)
            for i, row in enumerate(rows):
                # Replace only the two literal demonstration paths in published code.
                code = row['reference_code'].replace("'fixture.root'", repr(str(path))).replace(
                    "'answer.root'", repr(str(root / f'answer-{i}.root')))
                env = {}
                exec(compile(code, row['id'], 'exec'), env)
                self.assertEqual(env['result'], row['expected_result'])
                if row['task_type'] == 'writing':
                    with self.assertRaises(FileExistsError):
                        exec(compile(code, row['id'], 'exec'), {})


class ComparisonTest(unittest.TestCase):
    def test_alignment_blinding_and_missing_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / 'refs.jsonl'
            refs.write_text(json.dumps(dict(id='q1', question='Question?', category='files_io',
                                            answer='Reference', required_facts=['Fact'])) + '\n')
            predictions = {}
            for name in ('base', 'epoch1', 'epoch2', 'epoch3'):
                path = root / f'{name}.jsonl'
                path.write_text(json.dumps(dict(record_type='completion', id='q1',
                                                prompt='Question?', output=name)) + '\n')
                predictions[name] = path
            assemble(refs, predictions, root)
            review = json.loads((root / 'review.json').read_text())[0]
            mapping = json.loads((root / 'private_mapping.json').read_text())['q1']
            for slot, name in mapping.items():
                self.assertEqual(review['answers'][slot], name)
                self.assertIsNone(review['scores'][slot]['correctness'])
            predictions['epoch2'].write_text('')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                assemble(refs, predictions, root)


if __name__ == '__main__':
    unittest.main()
