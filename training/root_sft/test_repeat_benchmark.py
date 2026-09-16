import json
from pathlib import Path
import shutil
import tempfile
import unittest

from training.root_sft.repeat_benchmark import REPO, compare, digest, prepare, validate


class RepeatBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=REPO/'artifacts/root-sft', prefix='repeat-test-')
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name)/'original'
        self.source.mkdir()
        self.target = Path(self.tmp.name)/'repeat'
        for name in ('manifest.json','prompts.jsonl','references.jsonl','answers.jsonl','status.json','runtime.json'):
            shutil.copyfile(REPO/'artifacts/root-sft/qwen-baseline-50-001'/name,self.source/name)

    def completed_repeat(self):
        prepare(self.source,self.target)
        for name in ('answers.jsonl','status.json','runtime.json'):
            shutil.copyfile(self.source/name,self.target/name)

    def test_frozen_inputs_and_refuse_overwrite(self):
        prepare(self.source,self.target)
        for name in ('manifest.json','prompts.jsonl','references.jsonl'):
            self.assertEqual(digest(self.source/name),digest(self.target/name))
        with self.assertRaises(FileExistsError):prepare(self.source,self.target)

    def test_same_answers_are_not_automatically_passes(self):
        self.completed_repeat()
        compare(self.source,self.target)
        summary=json.loads((self.target/'comparison.json').read_text())
        self.assertEqual(summary['identical'],50)
        self.assertEqual(summary['changed'],0)
        review=json.loads((self.target/'comparison_review.json').read_text())
        self.assertTrue(all(r['correctness'] is None for r in review))

    def test_one_changed_answer(self):
        self.completed_repeat()
        rows=[json.loads(l) for l in (self.target/'answers.jsonl').read_text().splitlines()]
        next(r for r in rows if r['record_type']=='completion')['output'] += '\nChanged answer.'
        (self.target/'answers.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        status=json.loads((self.target/'status.json').read_text())
        status['answers_sha256']=digest(self.target/'answers.jsonl')
        (self.target/'status.json').write_text(json.dumps(status))
        compare(self.source,self.target)
        self.assertEqual(json.loads((self.target/'comparison.json').read_text())['changed'],1)

    def test_reject_input_tampering(self):
        prepare(self.source,self.target)
        with (self.target/'prompts.jsonl').open('a') as out:out.write('\n')
        with self.assertRaises(ValueError):validate(self.target)

    def test_reject_wrong_runtime_model(self):
        self.completed_repeat()
        runtime=json.loads((self.target/'runtime.json').read_text())
        runtime['model']='wrong/model'
        (self.target/'runtime.json').write_text(json.dumps(runtime))
        with self.assertRaises(ValueError):compare(self.source,self.target)


if __name__=='__main__':unittest.main()
