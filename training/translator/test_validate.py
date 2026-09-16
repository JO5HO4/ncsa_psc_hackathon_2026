import unittest

from training.translator.records import from_simple_qa
from training.translator.task_profiles import get_task_profile
from training.translator.translate import translate_record
from training.translator.validate import validate_output_rows, validate_source_records


class ValidateSourceRecordsTest(unittest.TestCase):
    def test_rejects_duplicate_id(self):
        records = [
            from_simple_qa({"id": "a", "question": "Q1", "answer": "A1"}),
            from_simple_qa({"id": "a", "question": "Q2", "answer": "A2"}),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            validate_source_records(records)

    def test_rejects_empty_question(self):
        record = from_simple_qa({"id": "a", "question": "x", "answer": "A1"})
        object.__setattr__(record, "question", "   ")
        with self.assertRaisesRegex(ValueError, "empty question"):
            validate_source_records([record])

    def test_accepts_valid_records(self):
        records = [from_simple_qa({"id": "a", "question": "Q1", "answer": "A1"})]
        validate_source_records(records)  # must not raise


class ValidateOutputRowsTest(unittest.TestCase):
    def setUp(self):
        self.task = get_task_profile("root_command")
        from training.translator.model_profiles import get_model_profile
        self.model = get_model_profile("qwen3.5")

    def test_accepts_valid_row(self):
        source = from_simple_qa({"id": "a", "question": "Q", "answer": "root -l -b -q -e 'x'"})
        row = translate_record(source, self.task, self.model)
        validate_output_rows([row], self.task)  # must not raise

    def test_rejects_row_failing_contract(self):
        row = {"id": "a", "messages": [
            {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "not a root command"},
        ]}
        with self.assertRaisesRegex(ValueError, "a:"):
            validate_output_rows([row], self.task)

    def test_rejects_wrong_roles(self):
        row = {"id": "a", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}
        with self.assertRaisesRegex(ValueError, "expected system/user/assistant"):
            validate_output_rows([row], self.task)


if __name__ == "__main__":
    unittest.main()
