#!/usr/bin/env python3
"""Reproducible, category-balanced diagnostic of the ORIGINAL Qwen checkpoint."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from training.root_sft.prepare import BANK, SYSTEM, category, digest, normalized, prompt, topic

MODEL = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
REVISION = '2e1fd397ee46e1388853d2af2c993145b0f1098a'


def select(rows, assignments):
    """Do not open test answers: exclude entire sealed assignment groups."""
    test_groups = {r['group'] for r in assignments if r['split'] == 'test'}
    excluded = {r['id'] for r in assignments if r['group'] in test_groups}
    candidates = [r for r in rows if r['id'] not in excluded and r['review_status'] != 'rejected']
    rng = random.Random(20260909)
    rng.shuffle(candidates)
    selected, seen_topics, seen_answers = [], set(), set()
    for cat in sorted({category(r) for r in candidates}):
        start = len(selected)
        for difficulty, number in [('introductory', 3), ('intermediate', 4), ('advanced', 3)]:
            options = [r for r in candidates if category(r) == cat and r['difficulty'] == difficulty]
            chosen = 0
            for row in options:
                if topic(row) in seen_topics or normalized(row['answer']) in seen_answers:
                    continue
                selected.append(dict(row, category=cat))
                seen_topics.add(topic(row))
                seen_answers.add(normalized(row['answer']))
                chosen += 1
                if chosen == number:
                    break
        # Some subjects have few advanced topics: preserve topic diversity rather
        # than padding with variants or borrowing sealed test questions.
        for row in candidates:
            if len(selected) - start == 10:
                break
            if category(row) != cat or topic(row) in seen_topics or normalized(row['answer']) in seen_answers:
                continue
            selected.append(dict(row, category=cat))
            seen_topics.add(topic(row))
            seen_answers.add(normalized(row['answer']))
        if len(selected) - start != 10:
            raise ValueError(f'Insufficient unique non-test topics for {cat}')
    return sorted(selected, key=lambda r: (r['category'], r['id']))


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    bank = [json.loads(x) for x in BANK.read_text().splitlines()]
    assignment_file = REPO/'data/root_io/splits/root-sft-v1/assignments.jsonl'
    assignments = [json.loads(x) for x in assignment_file.read_text().splitlines()]
    manifest_file = assignment_file.parent/'manifest.json'
    original = json.loads(manifest_file.read_text())
    if digest(BANK) != original['source_sha256'] or digest(assignment_file) != original['files_sha256']['assignments.jsonl']:
        raise ValueError('Bank or assignments changed; audit split membership first')
    rows = select(bank, assignments)
    if len(rows) != 50:
        raise ValueError('Expected five categories of ten questions')
    output.mkdir(parents=True)
    for name, values in [('references.jsonl', rows), ('prompts.jsonl', [dict(id=r['id'], prompt=prompt(r)) for r in rows])]:
        (output/name).write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in values))
    (output/'manifest.json').write_text(json.dumps(dict(model=MODEL, revision=REVISION,
        seed=20260909, system_prompt=SYSTEM, max_new_tokens=1024, temperature=0,
        categories=dict(Counter(r['category'] for r in rows)), difficulties=dict(Counter(r['difficulty'] for r in rows)),
        bank_sha256=digest(BANK), assignments_sha256=digest(assignment_file),
        prompts_sha256=digest(output/'prompts.jsonl'), references_sha256=digest(output/'references.jsonl'),
        limitations=['Existing public Q&A: diagnostic, not a new generalization test or a model capability ceiling.',
                     'Sealed test groups excluded. Some diagnostic questions may have appeared in prior SFT training/validation.',
                     'No ROOT tool access; evaluates direct answers only. Reference answers are not expert-certified.',
                     '1024-token output limit differs from earlier 512-token comparison.']), indent=2)+'\n')
    print(json.dumps({'n': len(rows), 'categories': dict(Counter(r['category'] for r in rows))}, indent=2))


def run(directory):
    from huggingface_hub import snapshot_download
    import torch
    manifest = json.loads((directory/'manifest.json').read_text())
    for name in ('prompts', 'references'):
        if digest(directory/f'{name}.jsonl') != manifest[f'{name}_sha256']:
            raise ValueError('Benchmark inputs changed')
    if (directory/'answers.jsonl').exists() or (directory/'status.json').exists():
        raise FileExistsError('Refusing to overwrite a prior run; prepare a fresh directory')
    started = datetime.now(timezone.utc).isoformat()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one visible GPU required')
        (directory/'status.json').write_text(json.dumps(dict(status='running', started_utc=started))+'\n')
        path = snapshot_download(repo_id=MODEL, revision=REVISION,
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
        if (Path(path)/'lora_adapter').exists():
            raise ValueError('Baseline must not contain our trained adapter')
        runtime = dict(model=MODEL, revision=REVISION, resolved_path=path,
            packages={p: version(p) for p in ('torch', 'transformers', 'huggingface-hub')},
            gpu=torch.cuda.get_device_name(0), manifest_sha256=digest(directory/'manifest.json'))
        (directory/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
        with (directory/'inference.log').open('w') as log:
            subprocess.run([sys.executable, 'inference/run_prompts.py', '--model', path,
                '--prompts', str(directory/'prompts.jsonl'), '--output', str(directory/'answers.jsonl'),
                '--format', 'chat', '--device', 'cuda', '--temperature', '0', '--max-new-tokens', '1024',
                '--system-prompt', SYSTEM], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        refs = [json.loads(x) for x in (directory/'references.jsonl').read_text().splitlines()]
        raw = [json.loads(x) for x in (directory/'answers.jsonl').read_text().splitlines()]
        predictions = [r for r in raw if r.get('record_type') == 'completion']
        answers = {r['id']: r for r in predictions}
        if len(answers) != len(predictions) or set(answers) != {r['id'] for r in refs}:
            raise ValueError('Incomplete or duplicate predictions')
        review = []
        report = ['# Original Qwen2.5-Coder-1.5B-Instruct: 50-question diagnostic', '',
                  'Generated answers, not yet correctness-scored. No training or ROOT tool execution.', '']
        for r in refs:
            a = answers[r['id']]
            if a['prompt'] != prompt(r) or not a['output'].strip():
                raise ValueError('Wrong prompt or empty answer')
            review.append(dict(r, model_answer=a['output'], scores=dict(correctness=None, critical_error=None, notes='')))
            report.extend([f"## {r['id']} — {r['category']} / {r['difficulty']}", '', prompt(r), '',
                           '### Model answer', '', a['output'], '', '### Reference answer', '', r['answer'], ''])
        (directory/'review.json').write_text(json.dumps(review, indent=2)+'\n')
        (directory/'ANSWERS.md').write_text('\n'.join(report))
        (directory/'status.json').write_text(json.dumps(dict(status='complete', started_utc=started,
            ended_utc=datetime.now(timezone.utc).isoformat(), answers=len(answers), answers_sha256=digest(directory/'answers.jsonl')))+'\n')
        print('Completed 50 answers:', directory, flush=True)
    except Exception as exc:
        (directory/'status.json').write_text(json.dumps(dict(status='failed', error=str(exc), started_utc=started))+'\n')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run'])
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args.directory.resolve())
    else:
        run(args.directory.resolve())
