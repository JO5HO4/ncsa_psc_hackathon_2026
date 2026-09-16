#!/usr/bin/env python3
"""Compare saved epochs on validation only; never train or touch the test set."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from training.root_sft.evaluate import records
from training.root_sft.prepare import digest, prompt
from training.root_sft.preflight import verify_release


def assemble(refs_path, predictions, output):
    if set(predictions) != {'base', 'epoch1', 'epoch2', 'epoch3'}:
        raise ValueError('Expected base and exactly three epoch predictions')
    refs = records(refs_path)
    if not refs:
        raise ValueError('Empty validation references')
    models = {name: records(path, True) for name, path in predictions.items()}
    for name, rows in models.items():
        if set(rows) != set(refs):
            raise ValueError(f"{name}: incomplete prediction IDs")
        for key, row in rows.items():
            if row['prompt'] != prompt(refs[key]):
                raise ValueError(f"{name}/{key}: prompt mismatch")
    rng = random.Random(20260908)
    review, mapping = [], {}
    for key, ref in refs.items():
        names = list(models)
        rng.shuffle(names)
        mapping[key] = dict(zip('ABCD', names))
        review.append(dict(id=key, category=ref['category'], question=prompt(ref),
                           reference=ref['answer'], required_facts=ref['required_facts'],
                           answers={slot: models[name][key]['output'] for slot, name in mapping[key].items()},
                           scores={slot: {'correctness': None, 'critical_error': None, 'notes': ''}
                                   for slot in mapping[key]}))
    for filename, value in [('review.json', review), ('private_mapping.json', mapping),
                            ('inputs.json', {name: {'path': str(path), 'sha256': digest(path)}
                                             for name, path in {'references': refs_path, **predictions}.items()}),
                            ('coverage.json', dict(Counter(r['category'] for r in refs.values())))]:
        (output / filename).write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=REPO / 'data/root_io/splits/root-sft-v1')
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    preflight = json.loads((run / 'preflight.json').read_text())
    if digest(args.data / 'manifest.json') != preflight['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest differs from training run')
    verify_release(args.data)
    original_inputs = json.loads((run / 'review/inputs.json').read_text())
    for name in ('before', 'after'):
        if digest(run / f'{name}.jsonl') != original_inputs[f'{name}_sha256']:
            raise ValueError(f'{name} predictions differ from original paired review')
    for step in (111, 222, 333):
        if not (run / f'checkpoints/global_step_{step}/fsdp_config.json').is_file():
            raise ValueError(f'Missing expected checkpoint {step}; this script targets the 111-step/epoch run')
    output.mkdir(parents=True)
    (output / 'status.json').write_text('{"status": "running"}\n')
    system = (run / 'system_prompt.txt').read_text().strip()
    predictions = {'base': run / 'before.jsonl', 'epoch3': run / 'after.jsonl'}
    with (output / 'comparison.log').open('w') as log:
        for epoch, step in [(1, 111), (2, 222)]:
            target = output / f'epoch{epoch}_model'
            commands = [
                ['bash', 'inference/export_verl_checkpoint.sh', str(run / f'checkpoints/global_step_{step}'), str(target)],
                [sys.executable, 'inference/run_prompts.py', '--model', str(target), '--prompts',
                 str(args.data / 'validation_prompts.jsonl'), '--output', str(output / f'epoch{epoch}.jsonl'),
                 '--format', 'chat', '--device', 'cuda', '--temperature', '0', '--max-new-tokens', '512',
                 '--system-prompt', system]]
            for command in commands:
                print('Running:', ' '.join(command), flush=True)
                try:
                    subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
                except subprocess.CalledProcessError as exc:
                    (output / 'status.json').write_text(json.dumps({'status': 'failed',
                        'epoch': epoch, 'exit_code': exc.returncode, 'command': command}) + '\n')
                    raise
            predictions[f'epoch{epoch}'] = output / f'epoch{epoch}.jsonl'
    assemble(args.data / 'validation_references.jsonl', predictions, output)
    (output / 'status.json').write_text(json.dumps({'status': 'complete', 'questions': 46,
        'note': 'Base and epoch3 predictions reused from this same run; epoch1/2 newly generated. Scores require review.'}) + '\n')
    print(f'Complete: {output}', flush=True)


if __name__ == '__main__':
    main()
