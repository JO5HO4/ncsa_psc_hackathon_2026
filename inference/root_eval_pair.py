"""Paired inference only: freeze prompts and base revision; never execute answers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_completions(path, prompts):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    answers = [r for r in rows if r.get('record_type') == 'completion']
    assert [r['id'] for r in answers] == [p['id'] for p in prompts], 'Completion IDs differ'
    assert all(r['output'].strip() for r in answers), 'Empty completion'
    return answers


def main():
    from huggingface_hub import snapshot_download
    mode, output = sys.argv[1:3]
    baseline_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('/baseline')
    adapter_path = Path(sys.argv[4]) if len(sys.argv) > 4 else Path('/adapter')
    out = Path(output)
    runner = Path('/workspace/inference/run_prompts.py')
    runtime = runner.with_name('model_runtime.py')
    split = os.environ.get('EVAL_SPLIT', 'test')
    assert split in ('test', 'validation'), 'EVAL_SPLIT must be test or validation'
    source = Path(f'/workspace/data/datasets/atlas-open-data-sft-dataset/data/sft/{split}.jsonl')
    code_hashes = {p.name: digest(p) for p in (runner, runtime)}
    if mode == 'base':
        prompts = []
        for line in source.read_text().splitlines():
            record = json.loads(line)
            users = [m['content'] for m in record['messages'] if m['role'] == 'user']
            assert len(users) == 1
            prompts.append({'id': record['id'], 'api_family': record['api_family'], 'prompt': users[0]})
        assert len(prompts) == 560 and len({p['id'] for p in prompts}) == 560
        base_model = os.environ['BASE_MODEL']
        snapshot = snapshot_download(base_model, revision=os.environ.get('BASE_REVISION') or None)
        system_prompt = os.environ.get(
            'SYSTEM_PROMPT',
            'Return exactly one executable ROOT command. Do not add explanation.',
        )
        protocol = {
            'base_model': base_model, 'base_revision': Path(snapshot).name,
            'dataset_sha256': digest(source), 'code_sha256': code_hashes,
            'count': 560, 'max_new_tokens': 256, 'temperature': 0,
            'eval_split': split,
            'enable_thinking': False,
            'system_prompt': system_prompt,
        }
        prompt_text = ''.join(json.dumps(p) + '\n' for p in prompts)
        protocol['prompts_sha256'] = hashlib.sha256(prompt_text.encode()).hexdigest()
    elif mode == 'lora':
        baseline = baseline_path
        protocol = json.loads((baseline / 'protocol.json').read_text())
        prompt_text = (baseline / 'prompts.jsonl').read_text()
        assert hashlib.sha256(prompt_text.encode()).hexdigest() == protocol['prompts_sha256']
        assert code_hashes == protocol['code_sha256'], 'Inference code changed since baseline'
        prompts = [json.loads(line) for line in prompt_text.splitlines()]
        validate_completions(baseline / 'completions.jsonl', prompts)
        config = json.loads((adapter_path / 'adapter_config.json').read_text())
        declared_base = config.get('base_model_name_or_path', '')
        snapshot = snapshot_download(protocol['base_model'], revision=protocol['base_revision'])
        assert declared_base in (protocol['base_model'], snapshot), 'Adapter base differs from baseline'
    else:
        raise ValueError('mode must be base or lora')
    for name, text in [('prompts.jsonl', prompt_text), ('protocol.json', json.dumps(protocol, indent=2))]:
        with (out / name).open('x') as handle:
            handle.write(text)
    print(json.dumps({'mode': mode, 'protocol': protocol}), flush=True)
    command = [sys.executable, str(runner), '--model', snapshot if mode == 'base' else str(adapter_path),
               '--prompts', str(out / 'prompts.jsonl'), '--prompt-field', 'prompt', '--id-field', 'id',
               '--format', 'chat', '--no-enable-thinking', '--system-prompt', protocol['system_prompt'],
               '--device', 'cuda', '--temperature', str(protocol['temperature']),
               '--max-new-tokens', str(protocol['max_new_tokens']), '--output', str(out / 'completions.jsonl')]
    if mode == 'lora':
        command += ['--base-model', snapshot]
    subprocess.run(command, check=True)
    answers = validate_completions(out / 'completions.jsonl', prompts)
    (out / 'inference-summary.json').write_text(json.dumps({
        'mode': mode, 'completed': len(answers), 'generation_seconds': sum(r['elapsed_seconds'] for r in answers),
        'root_execution_scoring': 'not_run',
    }, indent=2))
    print(f'PASS: {len(answers)} nonempty completions. ROOT execution scoring is a separate step.')


if __name__ == '__main__':
    main()
