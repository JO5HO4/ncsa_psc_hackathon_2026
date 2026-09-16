from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import unittest

from training.root_sft.prepare import BANK, build, components, digest, jsonl, normalized, prompt
from training.root_sft.preflight import verify_release
from training.root_sft.evaluate import CRITERIA, prepare, report
from students.root_io.build_release import load_records


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="root-sft-test-")
        cls.out = Path(cls.temp.name)/"release"
        cls.manifest = build(BANK, cls.out, allow_draft=True)
        cls.rows = load_records()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_source_preserved_and_artifacts_valid(self):
        import pyarrow.parquet as pq
        self.assertEqual(verify_release(self.out)["source_records"], 617)
        total = 0
        for split in ("train", "validation", "test"):
            rows = pq.read_table(self.out/f"{split}.parquet").to_pylist()
            total += len(rows)
            self.assertEqual(rows, [json.loads(l) for l in (self.out/f"{split}.jsonl").read_text().splitlines()])
            for r in rows:
                original = next(x for x in self.rows if x['id'] == r['id'])
                self.assertEqual([m['role'] for m in r['messages']], ["system", "user", "assistant"])
                self.assertEqual(r['messages'][1]['content'], prompt(original))
                self.assertEqual(r['messages'][2]['content'], original['answer'])
                self.assertEqual(json.loads(r['tools']), [])
                self.assertNotIn('required_facts', r['messages'][1]['content'])
        self.assertEqual(total, self.manifest['selected_records'])

    def test_no_known_link_crosses_splits(self):
        assignments = [json.loads(l) for l in (self.out/"assignments.jsonl").read_text().splitlines()]
        split = {r['id']: r['split'] for r in assignments}
        self.assertEqual(len(split), 617)
        for members in components(self.rows).values():
            self.assertEqual(len({split[r['id']] for r in members}), 1)

    def test_balance_and_global_deduplication(self):
        rows = [json.loads(l) for s in ("train", "validation", "test") for l in (self.out/f"{s}.jsonl").read_text().splitlines()]
        self.assertLessEqual(max(Counter(r['topic'] for r in rows).values()), 2)
        answers = [normalized(r['messages'][-1]['content']) for r in rows]
        self.assertEqual(len(answers), len(set(answers)))
        for s in ("train", "validation", "test"):
            subset = [r for r in rows if r['split'] == s]
            self.assertEqual(len({r['category'] for r in subset}), 5)
            self.assertEqual(len({r['difficulty'] for r in subset}), 3)

    def test_reproducible(self):
        other = Path(self.temp.name)/"repeated"
        m = build(BANK, other, allow_draft=True)
        self.assertEqual(self.manifest, m)
        self.assertEqual(digest(other/"manifest.json"), digest(self.out/"manifest.json"))

    def test_draft_gate_and_no_overwrite(self):
        with self.assertRaisesRegex(ValueError, "Draft"):
            build(BANK, Path(self.temp.name)/"draft")
        with self.assertRaises(FileExistsError):
            build(BANK, self.out, allow_draft=True)

    def test_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)/"release"
            build(BANK, out, allow_draft=True)
            (out/"train.jsonl").write_text("modified")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_release(out)


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="root-eval-test-")
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.refs = [dict(r, category="files_io") for r in load_records()[:2]]
        self.refpath = self.out/"refs.jsonl"
        jsonl(self.refpath, self.refs)
        self.predictions = [{"record_type": "completion", "id": r['id'], "prompt": prompt(r), "output": "Example answer"} for r in self.refs]
        for name in ("before", "after"):
            jsonl(self.out/f"{name}.jsonl", self.predictions)

    def make_review(self):
        return prepare(self.refpath, self.out/"before.jsonl", self.out/"after.jsonl", self.out/"review")

    def test_missing_predictions_rejected(self):
        jsonl(self.out/"after.jsonl", self.predictions[:1])
        with self.assertRaisesRegex(ValueError, "IDs"):
            self.make_review()

    def test_prompt_mismatch_rejected(self):
        self.predictions[0]['prompt'] = "wrong context"
        jsonl(self.out/"after.jsonl", self.predictions)
        with self.assertRaisesRegex(ValueError, "Prompt mismatch"):
            self.make_review()

    def test_unscored_answers_are_not_automatically_correct(self):
        self.make_review()
        with self.assertRaisesRegex(ValueError, "human scores"):
            report(self.out/"review/paired_review.jsonl", self.out/"review/private_mapping.json")

    def test_paired_report(self):
        rows = self.make_review()
        mapping = json.loads((self.out/"review/private_mapping.json").read_text())
        for r in rows:
            for slot in ("A", "B"):
                score = 2 if mapping[r['id']][slot] == "after" else 1
                r['scores'][slot] = {**dict.fromkeys(CRITERIA, score), "critical_error": False, "notes": "test"}
        jsonl(self.out/"scored.jsonl", rows)
        result = report(self.out/"scored.jsonl", self.out/"review/private_mapping.json")
        self.assertEqual(result['paired_mean_delta'], .5)
        self.assertEqual(result['improved'], 2)


@unittest.skipUnless(importlib.util.find_spec("safetensors") and importlib.util.find_spec("torch"), "Optional tensor dependencies unavailable")
class AdapterTest(unittest.TestCase):
    def test_zero_update_rejected_nonzero_accepted(self):
        import torch
        from safetensors.torch import save_file
        from training.root_sft.check_adapter import check
        with tempfile.TemporaryDirectory() as t:
            export = Path(t)
            adapter = export/"lora_adapter"
            adapter.mkdir()
            (adapter/"adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2}))
            state = {"model.q_proj.lora_A.weight": torch.ones(2, 3), "model.q_proj.lora_B.weight": torch.zeros(4, 2)}
            save_file(state, adapter/"adapter_model.safetensors")
            with self.assertRaisesRegex(ValueError, "No nonzero"):
                check(export)
            state["model.q_proj.lora_B.weight"] += 1
            save_file(state, adapter/"adapter_model.safetensors")
            self.assertEqual(check(export)['nonzero_layers'], 1)


class ShellTest(unittest.TestCase):
    def test_full_run_logging_success_and_failure(self):
        logger = Path(__file__).with_name('logging.sh').resolve()
        for code in (0, 7):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(['bash', '-c',
                    'set -euo pipefail\nsource "$LOGGER"\nroot_stage test_stage\necho stdout-marker\necho stderr-marker >&2\nexit "$TEST_EXIT"'],
                    env={**os.environ, 'LOGGER': str(logger), 'RUN_DIR': directory, 'TEST_EXIT': str(code)},
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, code, result.stderr)
                log = (Path(directory)/'run.log').read_text()
                self.assertIn('stdout-marker', log)
                self.assertIn('stderr-marker', log)
                self.assertIn(f'exit_code={code}', log)
                status = json.loads((Path(directory)/'status.json').read_text())
                self.assertEqual(status['exit_code'], code)
                self.assertEqual(status['stage'], 'test_stage')
                self.assertEqual(status['status'], 'success' if code == 0 else 'failed')

    def test_actual_wrapper_accepts_newline_free_tracker(self):
        script = Path(__file__).with_name("run.sh").read_text()
        block = script[script.index('STEP=""'):script.index('EXPORT_DIR=')]
        with tempfile.TemporaryDirectory() as directory:
            tracker = Path(directory)/"latest_checkpointed_iteration.txt"
            tracker.write_text("2")
            result = subprocess.run(["bash", "-c", "set -euo pipefail\n"+block],
                env={**os.environ, "SAVE_DIR": directory}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            tracker.write_text("not-a-step")
            result = subprocess.run(["bash", "-c", "set -euo pipefail\n"+block],
                env={**os.environ, "SAVE_DIR": directory}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
