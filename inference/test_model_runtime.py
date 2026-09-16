"""Model-loading contract checks without downloading a model or using a GPU."""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from inference.model_runtime import load_model


class ModelRuntimeTest(unittest.TestCase):
    def test_modern_dtype_keyword_and_inference_mode(self):
        for device in ('cpu', 'cuda'):
            with self.subTest(device=device):
                torch = types.ModuleType('torch')
                torch.float32, torch.float16, torch.bfloat16 = 'fp32', 'fp16', 'bf16'
                torch.cuda = MagicMock()
                torch.cuda.is_available.return_value = True
                torch.cuda.is_bf16_supported.return_value = True
                transformers = types.ModuleType('transformers')
                transformers.AutoTokenizer = MagicMock()
                transformers.AutoModelForCausalLM = MagicMock()
                tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
                tokenizer.pad_token_id = 0
                with patch.dict(sys.modules, torch=torch, transformers=transformers):
                    model, _, resolved, selected = load_model('Qwen/test-model', device)
                transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
                    'Qwen/test-model', dtype='fp32' if device == 'cpu' else 'bf16', trust_remote_code=False)
                model.to.assert_called_once_with(device)
                model.eval.assert_called_once_with()
                self.assertEqual(selected, device)


if __name__ == '__main__':
    unittest.main()
