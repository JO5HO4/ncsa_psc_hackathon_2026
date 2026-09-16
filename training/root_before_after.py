"""One allocation: baseline inference, LoRA SFT/export, paired inference.

No generated commands are executed and no test answers are used for training.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def run_stage(name, command, env, output):
    status = {'stage': name, 'started': datetime.now(timezone.utc).isoformat(), 'command': command}
    path = output / f'{name}-status.json'
    path.write_text(json.dumps(status, indent=2))
    print(f'START {name}; log: {output / (name + ".log")}', flush=True)
    with (output / f'{name}.log').open('x') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    status.update(exit_code=result.returncode, finished=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(status, indent=2))
    if result.returncode:
        raise RuntimeError(f'{name} failed with exit {result.returncode}; inspect its log')
    print(f'PASS {name}', flush=True)


def main():
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    import torch

    out = Path(sys.argv[1]).resolve()
    expected_gpus = int(os.environ['NPROC_PER_NODE'])
    assert torch.cuda.device_count() == expected_gpus, f'This profile requires {expected_gpus} visible GPUs'
    root = Path('/workspace')
    canonical_data = root / 'data/datasets/atlas-open-data-sft-dataset/data/sft'
    records = {split: pq.read_table(canonical_data / f'{split}.parquet', columns=['id']).column('id').to_pylist()
               for split in ('train', 'validation', 'test')}
    assert [len(records[s]) for s in records] == [4480, 560, 560]
    assert all(len(ids) == len(set(ids)) for ids in records.values())
    assert not (set(records['train']) & set(records['test']))
    assert not (set(records['validation']) & set(records['test']))
    assert not (set(records['train']) & set(records['validation']))
    # Freeze training/validation inputs as well as evaluation inputs.
    frozen = out / 'data'
    frozen.mkdir()
    import shutil
    hashes = {}
    sources = {
        'train': Path(os.environ.get('SOURCE_TRAIN_FILE', canonical_data / 'train.parquet')),
        'validation': Path(os.environ.get('SOURCE_VAL_FILE', canonical_data / 'validation.parquet')),
    }
    for split in ('train', 'validation'):
        if not sources[split].is_file():
            raise FileNotFoundError(f"Training input missing: {sources[split]}")
        destination = frozen / f'{split}.parquet'
        shutil.copyfile(sources[split], destination)
        hashes[split] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (out / 'data-manifest.json').write_text(json.dumps({'counts': {s: len(v) for s, v in records.items()}, 'sha256': hashes}, indent=2))
    baseline, after = out / 'base', out / 'lora'
    baseline.mkdir()
    after.mkdir()
    environment = os.environ.copy()
    inference_env = environment.copy()
    inference_env['CUDA_VISIBLE_DEVICES'] = environment.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')[0]
    pair = out / 'root_eval_pair.py'
    run_stage('01-base', [sys.executable, str(pair), 'base', str(baseline)], inference_env, out)
    protocol = json.loads((baseline / 'protocol.json').read_text())
    split = environment.get('EVAL_SPLIT', 'test')
    assert [p['id'] for p in map(json.loads, (baseline / 'prompts.jsonl').read_text().splitlines())] == records[split]
    snapshot = snapshot_download(protocol['base_model'], revision=protocol['base_revision'])
    training_env = environment.copy()
    settings = {
        'QWEN35_MODEL_SIZE': environment['QWEN35_MODEL_SIZE'], 'MODEL_PATH': snapshot,
        'NPROC_PER_NODE': str(expected_gpus),
        'TRAIN_FILE': str(frozen / 'train.parquet'), 'VAL_FILE': str(frozen / 'validation.parquet'),
        'SAVE_DIR': str(out / 'checkpoints'), 'TOTAL_EPOCHS': environment.get('TOTAL_EPOCHS', '1'),
        'TRAIN_BATCH_SIZE': '16', 'MICRO_BATCH_SIZE_PER_GPU': '1', 'MAX_LENGTH': '2048',
        'MAX_TOKEN_LEN_PER_GPU': '8192', 'LR': '1e-5', 'LORA_RANK': '16', 'LORA_ALPHA': '16',
        'RESUME_MODE': 'disable', 'SAVE_FREQ': 'after_each_epoch', 'MAX_CKPT_TO_KEEP': '1',
        'EXPORT_FOR_INFERENCE': 'true', 'EXPERIMENT_NAME': out.name,
    }
    training_env.update(settings)
    for key in ('LR', 'TOTAL_EPOCHS', 'TRAIN_BATCH_SIZE', 'MICRO_BATCH_SIZE_PER_GPU',
                'MAX_LENGTH', 'MAX_TOKEN_LEN_PER_GPU', 'LORA_RANK', 'LORA_ALPHA',
                'MAX_CKPT_TO_KEEP'):
        settings[key] = environment.get(key, settings[key])
    settings['TEST_FREQ'] = 'after_each_epoch'
    training_env.update(settings)
    (out / 'training-settings.json').write_text(json.dumps(settings, indent=2))
    recovery = environment.get('RECOVERY_CHECKPOINT', '')
    if recovery:
        checkpoint = Path(recovery)
        run_stage('02-export', [sys.executable, str(root / 'inference/export_verl_lora_adapter.py'),
                               '--checkpoint', str(checkpoint), '--base-model', snapshot], training_env, out)
        (out / 'recovery-source.json').write_text(json.dumps({'checkpoint': recovery}, indent=2))
    else:
        run_stage('02-train-export', ['bash', str(root / 'training/scripts/run_verl_sft.sh')], training_env, out)
        step = (out / 'checkpoints/latest_checkpointed_iteration.txt').read_text().strip()
        assert step.isdigit(), 'Invalid final checkpoint step'
        checkpoint = out / f'checkpoints/global_step_{step}'
    adapter = checkpoint / 'huggingface/lora_adapter'
    assert (adapter / 'adapter_config.json').is_file()
    assert (adapter / 'adapter_model.safetensors').stat().st_size > 0
    run_stage('03-lora', [sys.executable, str(pair), 'lora', str(after), str(baseline), str(adapter)], inference_env, out)
    assert (baseline / 'prompts.jsonl').read_bytes() == (after / 'prompts.jsonl').read_bytes()
    (out / 'pipeline-summary.json').write_text(json.dumps({
        'status': 'completed', 'eval_split': split, 'questions_per_model': 560,
        'base': str(baseline), 'lora': str(after), 'adapter': str(adapter),
        'root_execution_scoring': 'not_run',
    }, indent=2))
    print('PASS: baseline, LoRA training/export, and paired inference completed. ROOT scoring remains separate.')


if __name__ == '__main__':
    main()
