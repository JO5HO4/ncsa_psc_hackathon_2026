# Published model registry

The compact, reusable model artifacts are LoRA adapters published in
[`ho22joshua/hep-posttraining`](https://huggingface.co/ho22joshua/hep-posttraining).
Load each adapter with the indicated Qwen base model through PEFT; the
repository does not publish a merged base model.

| Dataset | Base model | LoRA configuration | Published directory |
| --- | --- | --- | --- |
| ROOT Dataset | Qwen3.5-0.8B | rank 32, 5 epochs, learning rate `1e-5` | `ROOT/qwen3.5-0.8b-lora-r32-e5-lr1e-5` |
| ROOT Dataset | Qwen3.5-4B | rank 32, 1 epoch, learning rate `1e-4` | `ROOT/qwen3.5-4b-lora-r32-e1-lr1e-4` |
| ROOT Dataset | Qwen3.5-9B | rank 32, 1 epoch, learning rate `1e-4` | `ROOT/qwen3.5-9b-lora-r32-e1-lr1e-4` |
| ROOT Dataset | Qwen3.5-9B | rank 32, 10 epochs, learning rate `1e-5` | `ROOT/qwen3.5-9b-lora-r32-e10-lr1e-5` |
| ROOT + TRExFitter Dataset | Qwen3.5-0.8B | rank 16, 3 epochs, learning rate `1e-5` | `ROOT-plus-TRExFitter/qwen3.5-0.8b-lora-r16-e3-lr1e-5` |
| ROOT + TRExFitter Dataset | Qwen3.5-0.8B | rank 32, 3 epochs, learning rate `1e-4` | `ROOT-plus-TRExFitter/qwen3.5-0.8b-lora-r32-e3-lr1e-4` |

Each published directory includes at least `adapter_model.safetensors`,
`adapter_config.json`, tokenizer files, and the chat template where applicable.
The 4B and 9B one-epoch ROOT adapter weights were hash-verified against their
local exports on 2026-10-12.

The active ROOT execution-feedback RL pilot is not yet a published model. It
will be added only after its compact adapter passes loading and held-out ROOT
6.40.02 verification.
