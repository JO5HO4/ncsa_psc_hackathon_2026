import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training import pipeline_runner as pr


class EnvFlagTest(unittest.TestCase):
    def test_parses_true_like_values(self):
        for value in ("true", "True", "1", "yes", "on"):
            with patch.dict(os.environ, {"X": value}, clear=False):
                self.assertTrue(pr.env_flag("X"))

    def test_parses_false_like_values(self):
        for value in ("false", "0", "no", ""):
            with patch.dict(os.environ, {"X": value}, clear=False):
                self.assertFalse(pr.env_flag("X"))

    def test_uses_default_when_unset(self):
        os.environ.pop("DOES_NOT_EXIST", None)
        self.assertFalse(pr.env_flag("DOES_NOT_EXIST"))
        self.assertTrue(pr.env_flag("DOES_NOT_EXIST", default=True))


class ValidateConfigTest(unittest.TestCase):
    def base_config(self, **overrides):
        config = dict(
            run_baseline_inference=True, run_training=True, run_post_training_inference=True,
            run_root_scoring=False, task_profile="root_command", existing_model=None,
        )
        config.update(overrides)
        return config

    def test_accepts_valid_config(self):
        pr.validate_config(self.base_config())  # must not raise

    def test_rejects_all_stages_off(self):
        config = self.base_config(
            run_baseline_inference=False, run_training=False, run_post_training_inference=False,
        )
        with self.assertRaisesRegex(ValueError, "At least one of"):
            pr.validate_config(config)

    def test_rejects_post_training_inference_without_training_or_existing_model(self):
        config = self.base_config(run_training=False, existing_model=None)
        with self.assertRaisesRegex(ValueError, "EXISTING_MODEL"):
            pr.validate_config(config)

    def test_accepts_post_training_inference_with_existing_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            pr.validate_config(self.base_config(run_training=False, existing_model=tmp))  # must not raise

    def test_rejects_root_scoring_for_non_executable_task_profile(self):
        config = self.base_config(run_root_scoring=True, task_profile="not_a_real_profile")
        with self.assertRaisesRegex(ValueError, "executable task profile"):
            pr.validate_config(config)

    def test_accepts_root_scoring_for_root_command(self):
        pr.validate_config(self.base_config(run_root_scoring=True, task_profile="root_command"))  # must not raise


class ResolveBaseModelTest(unittest.TestCase):
    def test_model_path_override_wins(self):
        self.assertEqual(pr.resolve_base_model({"MODEL_PATH": "custom/model"}), "custom/model")

    def test_qwen35_model_size_mapping(self):
        self.assertEqual(pr.resolve_base_model({"QWEN35_MODEL_SIZE": "9b"}), "Qwen/Qwen3.5-9B")
        self.assertEqual(pr.resolve_base_model({"QWEN35_MODEL_SIZE": "0.8b"}), "Qwen/Qwen3.5-0.8B")

    def test_rejects_unknown_size(self):
        with self.assertRaises(ValueError):
            pr.resolve_base_model({"QWEN35_MODEL_SIZE": "70b"})


class ExtractPromptsTest(unittest.TestCase):
    def test_extracts_single_user_message_per_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_file = Path(tmp) / "eval.jsonl"
            eval_file.write_text("".join(json.dumps(row) + "\n" for row in [
                {"id": "a", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "Q1"},
                                          {"role": "assistant", "content": "A1"}]},
                {"id": "b", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "Q2"},
                                          {"role": "assistant", "content": "A2"}]},
            ]))
            output = Path(tmp) / "prompts.jsonl"
            prompts = pr.extract_prompts(eval_file, output)
            self.assertEqual(prompts, [{"id": "a", "prompt": "Q1"}, {"id": "b", "prompt": "Q2"}])
            self.assertEqual(json.loads(output.read_text().splitlines()[0]), {"id": "a", "prompt": "Q1"})

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_file = Path(tmp) / "eval.jsonl"
            row = {"id": "a", "messages": [{"role": "user", "content": "Q"}]}
            eval_file.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate ids"):
                pr.extract_prompts(eval_file, Path(tmp) / "out.jsonl")

    def test_rejects_wrong_user_message_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_file = Path(tmp) / "eval.jsonl"
            eval_file.write_text(json.dumps({"id": "a", "messages": [{"role": "system", "content": "s"}]}) + "\n")
            with self.assertRaisesRegex(ValueError, "expected exactly one user message"):
                pr.extract_prompts(eval_file, Path(tmp) / "out.jsonl")

    def test_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_file = Path(tmp) / "eval.jsonl"
            eval_file.write_text(json.dumps({"id": "a", "messages": [{"role": "user", "content": "Q"}]}) + "\n")
            output = Path(tmp) / "out.jsonl"
            output.write_text("existing")
            with self.assertRaises(FileExistsError):
                pr.extract_prompts(eval_file, output)


class RunStageTest(unittest.TestCase):
    def test_writes_status_and_raises_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch.object(pr.subprocess, "run", return_value=type("Result", (), {"returncode": 17})()):
                with self.assertRaisesRegex(RuntimeError, "failed with exit 17"):
                    pr.run_stage("mystage", ["echo", "hi"], {}, output)
            status = json.loads((output / "mystage-status.json").read_text())
            self.assertEqual(status["exit_code"], 17)

    def test_success_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch.object(pr.subprocess, "run", return_value=type("Result", (), {"returncode": 0})()):
                pr.run_stage("mystage", ["echo", "hi"], {}, output)
            status = json.loads((output / "mystage-status.json").read_text())
            self.assertEqual(status["exit_code"], 0)


class MainIntegrationTest(unittest.TestCase):
    """Exercises main()'s stage-skip wiring with all subprocess/network calls
    mocked out -- no GPU/Perlmutter allocation needed to verify this logic."""

    @staticmethod
    def write_eval_file(path, ids):
        rows = [
            {"id": i, "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": f"Q{i}"},
                                    {"role": "assistant", "content": f"A{i}"}]}
            for i in ids
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_skips_training_and_uses_existing_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_file = tmp_path / "eval.jsonl"
            self.write_eval_file(eval_file, ["a", "b"])
            existing_model = tmp_path / "existing-model"
            existing_model.mkdir()
            run_dir = tmp_path / "run"
            run_dir.mkdir()

            env = {
                "TASK_PROFILE": "root_command", "RUN_BASELINE_INFERENCE": "false",
                "RUN_TRAINING": "false", "RUN_POST_TRAINING_INFERENCE": "true",
                "RUN_ROOT_SCORING": "false", "EXISTING_MODEL": str(existing_model),
                "QWEN35_MODEL_SIZE": "9b", "EVAL_FILE": str(eval_file),
            }
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(pr, "resolve_base_snapshot", return_value="/fake/base-snapshot"), \
                    patch.object(pr, "run_stage") as mock_run_stage, \
                    patch("sys.argv", ["pipeline_runner.py", str(run_dir)]):
                pr.main()

            stage_names = [call.args[0] for call in mock_run_stage.call_args_list]
            self.assertEqual(stage_names, ["trained-inference"])
            command = mock_run_stage.call_args_list[0].args[1]
            self.assertIn(str(existing_model), command)
            self.assertIn("/fake/base-snapshot", command)  # --base-model, for an adapter-only export

            summary = json.loads((run_dir / "pipeline-summary.json").read_text())
            self.assertIsNone(summary["export_dir"])
            self.assertEqual(summary["existing_model"], str(existing_model))
            self.assertEqual(summary["prompt_count"], 2)

    def test_training_stage_resolves_export_dir_from_tracker(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            run_dir.mkdir()

            def fake_run_stage(name, command, env, output):
                if name == "train":
                    save_dir = Path(env["SAVE_DIR"])
                    (save_dir / "global_step_42" / "huggingface").mkdir(parents=True)
                    (save_dir / "latest_checkpointed_iteration.txt").write_text("42")

            env = {
                "TASK_PROFILE": "root_command", "RUN_BASELINE_INFERENCE": "false",
                "RUN_TRAINING": "true", "RUN_POST_TRAINING_INFERENCE": "false",
                "RUN_ROOT_SCORING": "false", "QWEN35_MODEL_SIZE": "9b",
                "TRAIN_FILE": "train.parquet", "VAL_FILE": "val.parquet",
            }
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(pr, "resolve_base_snapshot", return_value="/fake/base-snapshot"), \
                    patch.object(pr, "run_stage", side_effect=fake_run_stage), \
                    patch("sys.argv", ["pipeline_runner.py", str(run_dir)]):
                pr.main()

            summary = json.loads((run_dir / "pipeline-summary.json").read_text())
            self.assertTrue(summary["export_dir"].endswith("global_step_42/huggingface"))

    def test_invalid_config_raises_before_any_stage_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            env = {"TASK_PROFILE": "root_command", "RUN_BASELINE_INFERENCE": "false",
                   "RUN_TRAINING": "false", "RUN_POST_TRAINING_INFERENCE": "false", "RUN_ROOT_SCORING": "false"}
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(pr, "run_stage") as mock_run_stage, \
                    patch("sys.argv", ["pipeline_runner.py", str(run_dir)]):
                with self.assertRaisesRegex(ValueError, "At least one of"):
                    pr.main()
            mock_run_stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
