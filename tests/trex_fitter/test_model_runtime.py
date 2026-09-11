import json

from inference.model_runtime import _adapter_base_model, resolve_model


def test_resolve_verl_adapter_only_export_and_base_model(tmp_path):
    export = tmp_path / "global_step_10" / "huggingface"
    adapter = export / "lora_adapter"
    adapter.mkdir(parents=True)
    (export / "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3.5-0.8B"}))
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": None}))

    assert resolve_model(export.parent) == str(export.resolve())
    assert _adapter_base_model(adapter, export) == "Qwen/Qwen3.5-0.8B"


def test_resolve_direct_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-0.8B"}))

    assert resolve_model(adapter) == str(adapter.resolve())
    assert _adapter_base_model(adapter, adapter) == "Qwen/Qwen3.5-0.8B"
