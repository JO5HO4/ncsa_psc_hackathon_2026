#!/usr/bin/env python3
"""Verify the exported default-LoRA checkpoint contains a learned weight update."""
import argparse
import json
from pathlib import Path


def check(export):
    from safetensors import safe_open
    import torch
    adapter = export/"lora_adapter"
    config = json.loads((adapter/"adapter_config.json").read_text())
    if config["r"] <= 0 or config["lora_alpha"] <= 0:
        raise ValueError("Invalid LoRA rank/scale in export")
    norms = {}
    with safe_open(adapter/"adapter_model.safetensors", framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for name in sorted(keys):
            if ".lora_B." not in name:
                continue
            a = f.get_tensor(name.replace(".lora_B.", ".lora_A.")).float()
            b = f.get_tensor(name).float()
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Nonfinite LoRA weights")
            # ||BA||_F^2 from small rank-by-rank Gram matrices; avoids dense weights.
            squared = float(torch.sum((b.T @ b) * (a @ a.T).T))
            norms[name] = max(squared, 0.) ** 0.5 * config["lora_alpha"] / config["r"]
    if not norms or not any(v > 0 for v in norms.values()):
        raise ValueError("No nonzero effective LoRA update found")
    return {"checked_layers": len(norms), "nonzero_layers": sum(v > 0 for v in norms.values()),
            "max_delta_frobenius_norm": max(norms.values()),
            "meaning": "Default zero-initialized LoRA-B now produces a nonzero weight delta; not proof of better answers."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("export", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = check(a.export)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
