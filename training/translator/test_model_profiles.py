import unittest

from training.translator.model_profiles import MODEL_PROFILES, get_model_profile


class ModelProfilesTest(unittest.TestCase):
    def test_qwen3_5_registered(self):
        profile = get_model_profile("qwen3.5")
        self.assertEqual(profile.tools_value, "[]")
        self.assertFalse(profile.enable_thinking)

    def test_unknown_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model profile"):
            get_model_profile("does-not-exist")

    def test_registry_lookup_matches_dict(self):
        self.assertIs(get_model_profile("qwen3.5"), MODEL_PROFILES["qwen3.5"])


if __name__ == "__main__":
    unittest.main()
