import unittest

from training.translator.model_profiles import get_model_profile
from training.translator.records import from_atlas_command_raw, from_simple_qa
from training.translator.task_profiles import get_task_profile
from training.translator.translate import render_split, translate_record


class TranslateRecordTest(unittest.TestCase):
    def setUp(self):
        self.task = get_task_profile("root_command")
        self.model = get_model_profile("qwen3.5")

    def test_simple_qa_round_trip(self):
        source = from_simple_qa({
            "id": "q1",
            "question": "Print 2+2.",
            "category": "mathematics",
            "answer": "root -l -b -q -e 'std::cout << \"RESULT=\" << 2+2 << std::endl; gSystem->Exit(0);'",
        })
        row = translate_record(source, self.task, self.model)
        self.assertEqual([m["role"] for m in row["messages"]], ["system", "user", "assistant"])
        self.assertEqual(row["messages"][0]["content"], self.task.system_prompt)
        self.assertEqual(row["messages"][1]["content"], "Print 2+2.")
        self.assertEqual(row["messages"][2]["content"], source.answer)
        self.assertEqual(row["tools"], "[]")
        self.assertIs(row["enable_thinking"], False)
        self.assertEqual(row["format_version"], "qwen-root-command-chat-v1")
        self.assertEqual(row["category"], "mathematics")

    def test_invalid_answer_rejected_before_write(self):
        source = from_simple_qa({"id": "q2", "question": "Q", "answer": "not a root command"})
        with self.assertRaises(ValueError):
            translate_record(source, self.task, self.model)

    def test_atlas_command_adapter_preserves_key_order_and_original_messages(self):
        raw = {
            "api_family": "mathematics",
            "execution": {"expected_result": "3", "result_line_count": 1, "returncode": 0, "status": "verified"},
            "ground_truth": "root -l -b -q -e 'std::cout << \"RESULT=\" << TMath::Sqrt(9.) << std::endl; gSystem->Exit(0);'",
            "id": "root_docs_generated_tmath_0002",
            "messages": [
                {"content": "placeholder system", "role": "system", "tool_call_id": None, "tool_calls": None},
                {"content": "How do you print TMath::Sqrt(9)?", "role": "user", "tool_call_id": None, "tool_calls": None},
                {
                    "content": "root -l -b -q -e 'std::cout << \"RESULT=\" << TMath::Sqrt(9.) << std::endl; gSystem->Exit(0);'",
                    "role": "assistant", "tool_call_id": None, "tool_calls": None,
                },
            ],
            "source": "tmath",
            "tools": "[]",
        }
        source = from_atlas_command_raw(raw)
        row = translate_record(source, self.task, self.model)
        self.assertEqual(
            list(row.keys()),
            ["api_family", "execution", "ground_truth", "id", "messages", "source", "tools",
             "enable_thinking", "format_version"],
        )
        self.assertEqual(row["messages"][0]["content"], self.task.system_prompt)
        # User/assistant messages are reused verbatim (same dict, same key order) from the raw record.
        self.assertEqual(list(row["messages"][1].keys()), ["content", "role", "tool_call_id", "tool_calls"])
        self.assertEqual(row["messages"][2], raw["messages"][2])

    def test_render_split_rejects_duplicate_ids(self):
        source = from_simple_qa({"id": "dup", "question": "Q", "answer": "root -l -b -q -e 'x'"})
        with self.assertRaisesRegex(ValueError, "Duplicate id"):
            render_split([source, source], self.task, self.model)


if __name__ == "__main__":
    unittest.main()
