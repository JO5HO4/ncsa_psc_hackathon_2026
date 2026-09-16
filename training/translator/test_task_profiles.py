import unittest

from training.translator.task_profiles import get_task_profile


class RootCommandTaskProfileTest(unittest.TestCase):
    def setUp(self):
        self.task = get_task_profile("root_command")

    def test_accepts_valid_command(self):
        self.task.validate_answer("root -l -b -q -e 'std::cout << \"RESULT=\" << 1 << std::endl;'")

    def test_rejects_missing_wrapper(self):
        with self.assertRaises(ValueError):
            self.task.validate_answer("std::cout << 1;")

    def test_rejects_missing_closing_quote(self):
        with self.assertRaises(ValueError):
            self.task.validate_answer("root -l -b -q -e 'std::cout << 1;")

    def test_rejects_multiline(self):
        with self.assertRaises(ValueError):
            self.task.validate_answer("root -l -b -q -e 'a'\nroot -l -b -q -e 'b'")

    def test_unknown_task_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown task profile"):
            get_task_profile("does-not-exist")


if __name__ == "__main__":
    unittest.main()
