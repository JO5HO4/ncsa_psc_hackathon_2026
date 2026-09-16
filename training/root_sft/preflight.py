#!/usr/bin/env python3
"""Fail-fast integrity/tokenization checks using the actual repository verl adapter."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from training.root_sft.prepare import BANK, digest


def verify_release(directory):
    manifest = json.loads((directory/"manifest.json").read_text())
    if digest(BANK) != manifest["source_sha256"]:
        raise ValueError("Source bank changed since preparation; build a new release")
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or digest(directory/name) != expected:
            raise ValueError(f"Dataset artifact checksum mismatch: {name}")
    return manifest


def run(data, output, model, revision, max_length, batch_size):
    import torch
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from training.verl_dataset import TReXConfigSFTDataset
    manifest = verify_release(data)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU preflight requires a Perlmutter GPU allocation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This one-GPU baseline expects exactly one visible CUDA device")
    path = str(Path(model).resolve()) if Path(model).is_dir() else snapshot_download(
        repo_id=model, revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    lengths = {}
    for split in ("train", "validation", "test"):
        config = OmegaConf.create({"messages_key": "messages", "tools_key": "tools",
            "max_length": max_length, "truncation": "error", "pad_mode": "no_padding",
            "ignore_input_ids_mismatch": False})
        ds = TReXConfigSFTDataset([str(data/f"{split}.parquet")], tokenizer, config)
        if split != "test" and (len(ds) < batch_size or len(ds) % batch_size):
            raise ValueError(f"{split}: {len(ds)} rows not divisible by batch {batch_size}; verl drops the last incomplete batch. Choose a compatible batch (default 2).")
        values = []
        for i in range(len(ds)):
            item = ds[i]
            if not 0 < int(item["loss_mask"].sum()) < item["input_ids"].numel():
                raise ValueError(f"Invalid assistant-only loss mask in {split} row {i}")
            prefix = tokenizer.apply_chat_template(ds.messages[i][:-1], tools=[], add_generation_prompt=True)
            if int(item["loss_mask"][:len(prefix)].sum()) != 0 or not bool(item["loss_mask"][len(prefix):].all()):
                raise ValueError(f"Loss mask supervises the wrong token span in {split} row {i}")
            values.append(item["input_ids"].numel())
        lengths[split] = {"rows": len(ds), "max_tokens": max(values), "min_tokens": min(values)}
    report = {"model_requested": model, "revision_requested": revision,
        "resolved_model_path": path, "model_config_sha256": digest(Path(path)/"config.json"),
        "dataset_manifest_sha256": digest(data/"manifest.json"), "token_lengths": lengths,
        "max_length": max_length, "batch_size": batch_size,
        "gpu": torch.cuda.get_device_name(0), "cuda_device_count": torch.cuda.device_count(),
        "container_image": os.environ.get("VERL_IMAGE"), "container_image_id": os.environ.get("VERL_IMAGE_ID"),
        "packages": {p: version(p) for p in ("torch", "transformers", "peft", "pyarrow", "verl")},
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "verl_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO/"verl", text=True).strip(),
        "system_prompt": manifest["system_prompt"]}
    report["pipeline_sha256"] = {str(p.relative_to(REPO)): digest(p) for p in sorted((REPO/"training/root_sft").glob("*")) if p.is_file()}
    (output/"preflight.json").write_text(json.dumps(report, indent=2)+"\n")
    (output/"model_path.txt").write_text(path+"\n")
    (output/"system_prompt.txt").write_text(manifest["system_prompt"]+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    a = p.parse_args()
    run(a.data.resolve(), a.output.resolve(), a.model, a.revision, a.max_length, a.batch_size)
