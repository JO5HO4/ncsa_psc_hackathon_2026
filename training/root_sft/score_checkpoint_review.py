#!/usr/bin/env python3
"""Validate explicit reviewer judgments and aggregate them; does not judge with a model."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

MODELS = {'base', 'epoch1', 'epoch2', 'epoch3'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annotations(path):
    lines = path.read_text().splitlines()
    if not lines or lines[0] != 'id\tA\tB\tC\tD':
        raise ValueError('Expected TSV header id A B C D')
    result = {}
    for line in lines[1:]:
        cells = line.split('\t')
        if len(cells) != 5:
            raise ValueError('Expected ID and four judgments')
        key = 'root_knowledge_' + cells[0]
        if key in result:
            raise ValueError('Duplicate annotation ID')
        result[key] = {}
        for slot, cell in zip('ABCD', cells[1:]):
            token, note = cell.split('|', 1)
            if token not in ('0', '1', '2', '0!', '1!') or not note.strip():
                raise ValueError('Expected 0/1/2, optional critical !, and a nonempty note')
            result[key][slot] = dict(correctness=int(token[0]), critical_error=token.endswith('!'), notes=note)
    return result


def metrics(scores):
    return dict(n=len(scores), mean_correctness=statistics.mean(s['correctness'] for s in scores),
                fully_correct=sum(s['correctness'] == 2 for s in scores),
                critical_errors=sum(s['critical_error'] for s in scores),
                critical_error_rate=statistics.mean(s['critical_error'] for s in scores),
                mean_critical_gated=statistics.mean(0 if s['critical_error'] else s['correctness'] for s in scores))


def aggregate(rows, mapping):
    if not rows or len({r['id'] for r in rows}) != len(rows) or {r['id'] for r in rows} != set(mapping):
        raise ValueError('Unique review and mapping IDs must agree')
    buckets = {m: defaultdict(list) for m in MODELS}
    by_id = {}
    for row in rows:
        key = row['id']
        if set(mapping[key]) != set('ABCD') or set(mapping[key].values()) != MODELS:
            raise ValueError('Each question must map to four different models')
        if set(row['scores']) != set('ABCD'):
            raise ValueError('Missing score slot')
        by_id[key] = {}
        for slot, model in mapping[key].items():
            score = row['scores'][slot]
            if type(score.get('correctness')) is not int or score['correctness'] not in (0, 1, 2):
                raise ValueError('Unscored or invalid correctness')
            if type(score.get('critical_error')) is not bool or not score.get('notes', '').strip():
                raise ValueError('Critical-error flag and note required')
            if score['critical_error'] and score['correctness'] == 2:
                raise ValueError('Fully correct cannot contain a critical error')
            buckets[model][row['category']].append(score)
            by_id[key][model] = score
    summary = {}
    for model, categories in sorted(buckets.items()):
        values = [s for scores in categories.values() for s in scores]
        cats = {c: metrics(scores) for c, scores in sorted(categories.items())}
        summary[model] = dict(overall=metrics(values), categories=cats,
            macro_category_correctness=statistics.mean(x['mean_correctness'] for x in cats.values()),
            macro_category_critical_rate=statistics.mean(x['critical_error_rate'] for x in cats.values()),
            macro_category_critical_gated=statistics.mean(x['mean_critical_gated'] for x in cats.values()))
    pairs = {}
    for first, second in [('base', 'epoch1'), ('base', 'epoch2'), ('base', 'epoch3'),
                          ('epoch1', 'epoch2'), ('epoch2', 'epoch3')]:
        delta = [r[second]['correctness'] - r[first]['correctness'] for r in by_id.values()]
        pairs[f'{first}_to_{second}'] = dict(improved=sum(d > 0 for d in delta), tied=delta.count(0),
            regressed=sum(d < 0 for d in delta), mean_correctness_delta=statistics.mean(delta),
            new_critical_errors=sum(not r[first]['critical_error'] and r[second]['critical_error'] for r in by_id.values()),
            resolved_critical_errors=sum(r[first]['critical_error'] and not r[second]['critical_error'] for r in by_id.values()))
    return dict(models=summary, paired=pairs)


def build(review, annotation_file, mapping_file, output):
    if output.exists():
        raise FileExistsError(output)
    rows = json.loads(review.read_text())
    judgments = annotations(annotation_file)
    if set(judgments) != {r['id'] for r in rows}:
        raise ValueError('Annotation coverage differs from review')
    for row in rows:
        row['scores'] = judgments[row['id']]
    # Judgments are authored and persisted BEFORE this mapping is loaded.
    mapping = json.loads(mapping_file.read_text())
    result = aggregate(rows, mapping)
    output.mkdir(parents=True)
    metadata = dict(reviewer='OpenAI assistant; no independent human expert review',
        method='Manual semantic judgments authored with A/B/C/D labels before reading mapping; prior exposure to some outputs means not fully blind.',
        created_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256=dict(review=sha(review), annotations=sha(annotation_file), mapping=sha(mapping_file)),
        rubric={'0': 'Core answer wrong or missing.', '1': 'Partially correct or contains material caveats/errors.',
                '2': 'Answers the question correctly without material false claims; not exact reference matching.',
                'critical_error': 'Central false concept, wrong formula, misleading analysis instruction, or materially invalid code. Omissions alone are not critical.',
                'critical_gated': 'Correctness becomes zero if a critical error is present; raw correctness is retained separately.'},
        limitations=['46 public knowledge validation prompts, 5-16 per category; related topics are not independent trials.',
                     'No execution of generated code and no independent expert certification.',
                     'No formal significance claim or unbiased generalization estimate; test set not used.',
                     'Scores cover displayed generations, including unsupported additions; verbosity can expose more errors.',
                     'Base and epoch3 outputs reused, epoch1/2 generated later; runtime differences are a possible confound.'],
        sources=['https://root.cern/manual/root_files/', 'https://root.cern/manual/object_ownership/',
                 'https://root.cern/doc/master/classTH1.html', 'https://root.cern/doc/master/classROOT_1_1RDataFrame.html',
                 'https://root.cern/doc/master/rf605__profilell_8py.html', 'https://root.cern/doc/master/classRooAbsPdf.html',
                 'https://root.cern/doc/master/namespaceRooStats.html', 'https://uproot.readthedocs.io/en/latest/basic.html'])
    result['review_metadata'] = metadata
    (output / 'scored_review.json').write_text(json.dumps(rows, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    report = ['# Assistant-scored checkpoint comparison', '',
              'Provisional semantic review, not expert certification or an executable benchmark.', '',
              'Correctness: 0 = wrong/missing, 1 = partial, 2 = correct. Critical flags are separate.', '',
              '| Model | Mean / 2 | Fully correct / 46 | Critical / 46 | Category macro / 2 |',
              '|---|---:|---:|---:|---:|']
    for model in ('base', 'epoch1', 'epoch2', 'epoch3'):
        s = result['models'][model]
        o = s['overall']
        report.append(f"| {model} | {o['mean_correctness']:.3f} | {o['fully_correct']} | {o['critical_errors']} | {s['macro_category_correctness']:.3f} |")
    report += ['', '## By category', '', 'Each cell: mean correctness / 2; critical-error count.', '',
               '| Category | n | Base | Epoch 1 | Epoch 2 | Epoch 3 |', '|---|---:|---:|---:|---:|---:|']
    for cat, c in result['models']['base']['categories'].items():
        cells = [f"{result['models'][m]['categories'][cat]['mean_correctness']:.2f}; {result['models'][m]['categories'][cat]['critical_errors']}" for m in ('base', 'epoch1', 'epoch2', 'epoch3')]
        report.append(f"| {cat} | {c['n']} | " + ' | '.join(cells) + ' |')
    report += ['', '## Limits and audit trail', '', metadata['method'], ''] + ['- ' + x for x in metadata['limitations']]
    report += ['', 'See scored_review.json for all answer-level rationales, summary.json for raw/gated metrics, paired changes and source links.', '']
    (output / 'REPORT.md').write_text('\n'.join(report))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('review', 'annotations', 'mapping', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.review, args.annotations, args.mapping, args.output)
    print(f'Scored review: {args.output}')
