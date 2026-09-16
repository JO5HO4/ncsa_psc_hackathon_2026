#!/usr/bin/env python3
"""Repeat frozen benchmark inputs and compare answers, without training.

An exact greedy repeat checks reproducibility, not independent-trial accuracy.
Changed answers require a fresh correctness review; equality is not correctness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(directory, completed=False):
    manifest = read(directory/'manifest.json')
    for name in ('prompts', 'references'):
        if digest(directory/f'{name}.jsonl') != manifest[f'{name}_sha256']:
            raise ValueError(f'Changed {name} input')
    if completed:
        status = read(directory/'status.json')
        if status['status'] != 'complete' or status['answers'] != 50:
            raise ValueError('Expected a completed 50-question run')
        if digest(directory/'answers.jsonl') != status['answers_sha256']:
            raise ValueError('Changed answer artifact')
        runtime = read(directory/'runtime.json')
        if runtime['manifest_sha256'] != digest(directory/'manifest.json'):
            raise ValueError('Runtime manifest mismatch')
        if any(runtime[k] != manifest[k] for k in ('model', 'revision')):
            raise ValueError('Runtime model does not match the manifest')
    return manifest


def prepare(source, target):
    manifest = validate(source, completed=True)
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    target.mkdir(parents=True)
    for name in ('manifest.json', 'prompts.jsonl', 'references.jsonl'):
        shutil.copyfile(source/name, target/name)
    # Store repository-relative provenance so host and container paths agree.
    (target/'repeat_source.json').write_text(json.dumps(dict(
        source=str(source.relative_to(REPO)),
        answers_sha256=digest(source/'answers.jsonl'),
        manifest_sha256=digest(source/'manifest.json'),
        purpose='Exact-input greedy reproducibility check, not independent trials',
        model=manifest['model']), indent=2)+'\n')
    print(f'Prepared the identical 50 questions in {target}')


def answers(directory):
    rows = [json.loads(l) for l in (directory/'answers.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['record_type'] == 'completion']
    result = {r['id']: r for r in rows}
    if len(result) != 50 or len(rows) != 50 or any(not r['output'].strip() for r in rows):
        raise ValueError('Missing, duplicate or empty answers')
    return result


def compare(source, target):
    old_manifest = validate(source, completed=True)
    if validate(target, completed=True) != old_manifest:
        raise ValueError('Not an identical-input/settings repeat')
    old, new = answers(source), answers(target)
    if old.keys() != new.keys():
        raise ValueError('Question IDs differ')
    refs = {r['id']: r for r in map(json.loads, (source/'references.jsonl').read_text().splitlines())}
    prompts = {r['id']: r['prompt'] for r in map(json.loads, (source/'prompts.jsonl').read_text().splitlines())}
    rows = []
    for key in old:
        if old[key]['prompt'] != new[key]['prompt'] or new[key]['prompt'] != prompts[key]:
            raise ValueError(f'Prompt mismatch: {key}')
        rows.append(dict(id=key, category=refs[key]['category'], question=refs[key]['question'],
            reference_answer=refs[key]['answer'], original_answer=old[key]['output'],
            repeat_answer=new[key]['output'], identical=old[key]['output'] == new[key]['output'],
            correctness=None, critical_error=None))
    old_runtime, new_runtime = read(source/'runtime.json'), read(target/'runtime.json')
    differences = {k: dict(original=old_runtime.get(k), repeat=new_runtime.get(k))
                   for k in ('model', 'revision', 'gpu', 'packages') if old_runtime.get(k) != new_runtime.get(k)}
    summary = dict(n=50, identical=sum(r['identical'] for r in rows),
        changed=sum(not r['identical'] for r in rows), runtime_differences=differences,
        by_category={c:dict(n=sum(r['category']==c for r in rows),
                           identical=sum(r['category']==c and r['identical'] for r in rows)) for c in sorted({r['category'] for r in rows})},
        meaning='Text agreement, not factual accuracy. Changed answers need review; identical errors remain errors.')
    (target/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    (target/'comparison_review.json').write_text(json.dumps(rows,indent=2)+'\n')
    (target/'RECHECK.md').write_text('# Same-50-question repeat\n\n'
        f"Identical answers: {summary['identical']}/50. Changed answers: {summary['changed']}/50.\n\n"
        'This is a greedy reproducibility check, not 50 new independent questions and not a correctness score. '
        'Review comparison_review.json for factual correctness; do not count repeated errors as successes.\n\n'
        'Runtime differences: `'+json.dumps(differences)+'`\n')
    print(json.dumps(summary,indent=2))


def run(target):
    from training.root_sft.quick_benchmark import MODEL, REVISION, run as infer
    manifest = validate(target)
    if (manifest['model'],manifest['revision'],manifest['temperature'],manifest['max_new_tokens']) != (MODEL,REVISION,0,1024):
        raise ValueError('Runner does not match frozen settings')
    origin = read(target/'repeat_source.json')
    source = REPO/origin['source']
    if digest(source/'answers.jsonl') != origin['answers_sha256'] or digest(source/'manifest.json') != origin['manifest_sha256']:
        raise ValueError('Original run changed after repeat preparation')
    infer(target)
    compare(source,target)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','compare'])
    p.add_argument('--source',type=Path,default=REPO/'artifacts/root-sft/qwen-baseline-50-001')
    p.add_argument('--directory',type=Path,required=True)
    a = p.parse_args()
    if a.command == 'prepare': prepare(a.source.resolve(),a.directory.resolve())
    elif a.command == 'run': run(a.directory.resolve())
    else: compare(a.source.resolve(),a.directory.resolve())
