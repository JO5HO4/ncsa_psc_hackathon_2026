#!/usr/bin/env python3
"""Balanced ROOT operation-plan SFT and executable held-out-fixture evaluation.

No model-produced Python, shell, expression, or arbitrary file path is executed.
This is a narrow single-operation tool-selection benchmark, not an autonomous agent.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

SYSTEM = 'Return one JSON object calling the documented ROOT operation. No prose, code fences or predicted results.'
FIELDS = {
    'class': ['object'], 'cycles': ['object'],
    'select_gt': ['tree', 'column', 'threshold', 'id_column'],
    'select_range': ['tree', 'column', 'low', 'high', 'id_column'],
    'jagged_count_gt': ['tree', 'column', 'threshold'], 'jagged_sum': ['tree', 'column'],
    'weighted_hist': ['tree', 'column', 'weight', 'edges'],
    'selected_yield': ['tree', 'column', 'threshold', 'weight'],
    'stream_sum': ['tree', 'column', 'threshold', 'weight', 'chunk_entries'],
    'check_counts': ['tree', 'column', 'count', 'id_column'],
    'write_skim': ['tree', 'column', 'threshold', 'id_column'],
    'write_hist': ['tree', 'column', 'edges'],
}
DESCRIPTIONS = {
    'class': 'Return the ROOT class name of object.',
    'cycles': 'Return stored string versions of object in increasing cycle order.',
    'select_gt': 'Return IDs where scalar column > threshold.',
    'select_range': 'Return IDs where low <= scalar column < high.',
    'jagged_count_gt': 'Return per-row counts of collection elements > threshold, retaining empty rows.',
    'jagged_sum': 'Return scalar sum of collection elements per row, zero for empty rows.',
    'weighted_hist': 'Return sumw and sumw2 in edges; NumPy bin convention, discard flow; independent entries.',
    'selected_yield': 'Return entries and sumw where column > threshold.',
    'stream_sum': 'Return sumw where column > threshold, reading chunk_entries rows per chunk.',
    'check_counts': 'Return IDs where count differs from per-row length of collection column.',
    'write_skim': 'Write new Selected TTree with id_column and column where column > threshold; reopen and return class and IDs.',
    'write_hist': 'Write new unweighted histogram of column with edges, reopen and return values and edges.',
}
CONTRACT = '\n'.join(f'{op}({", ".join(fields)}): {DESCRIPTIONS[op]}' for op, fields in FIELDS.items())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(plan, input_file, output_dir):
    """Only reads the harness-supplied trusted fixture; writes fixed new paths."""
    if not isinstance(plan, dict) or plan.get('operation') not in FIELDS:
        raise ValueError('Unknown operation')
    op = plan['operation']
    if set(plan) != {'operation', *FIELDS[op]}:
        raise ValueError('Unexpected or missing arguments; paths and code are not accepted')
    for key, value in plan.items():
        if key in ('threshold', 'low', 'high'):
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError('Expected finite numeric threshold')
        elif key == 'edges':
            if not isinstance(value, list) or not 2 <= len(value) <= 100:
                raise ValueError('Invalid edges')
            if any(type(x) not in (int, float) or not np.isfinite(x) for x in value) or any(a >= b for a, b in zip(value, value[1:])):
                raise ValueError('Edges must be finite and increasing')
        elif key == 'chunk_entries':
            if type(value) is not int or not 1 <= value <= 1000:
                raise ValueError('Invalid chunk size')
        elif not isinstance(value, str) or len(value) > 100:
            raise ValueError('Expected short object/branch name')
    with uproot.open(input_file) as f:
        if op == 'class':
            return f[plan['object']].classname
        if op == 'cycles':
            keys = [k for k in f.keys() if k.rsplit(';', 1)[0] == plan['object']]
            return [str(f[k]) for k in sorted(keys, key=lambda x: int(x.rsplit(';', 1)[1]))]
        tree = f[plan['tree']]
        col = plan['column']
        if op in ('jagged_count_gt', 'jagged_sum'):
            a = tree[col].array(library='ak')
            return ak.to_list(ak.sum(a > plan['threshold'] if op == 'jagged_count_gt' else a, axis=1))
        if op == 'check_counts':
            a = tree.arrays([col, plan['count'], plan['id_column']], library='ak')
            return ak.to_list(a[plan['id_column']][ak.num(a[col], axis=1) != a[plan['count']]])
        if op == 'stream_sum':
            total = 0.
            for a in tree.iterate([col, plan['weight']], step_size=plan['chunk_entries'], library='np'):
                total += float(a[plan['weight']][a[col] > plan['threshold']].sum())
            return total
        names = list(dict.fromkeys([col] + [plan[k] for k in ('id_column', 'weight') if k in plan]))
        a = tree.arrays(names, library='np')
        if op == 'select_range':
            return a[plan['id_column']][(a[col] >= plan['low']) & (a[col] < plan['high'])].tolist()
        if op in ('select_gt', 'selected_yield', 'write_skim'):
            mask = a[col] > plan['threshold']
            if op == 'select_gt':
                return a[plan['id_column']][mask].tolist()
            if op == 'selected_yield':
                return {'entries': int(mask.sum()), 'sumw': float(a[plan['weight']][mask].sum())}
            target = output_dir / 'skim.root'
            with uproot.create(target) as out:
                selected = {k: v[mask] for k, v in a.items()}
                out.mktree('Selected', {k: v.dtype for k, v in selected.items()}).extend(selected)
            with uproot.open(target) as out:
                return {'class': out['Selected'].classname, 'ids': out['Selected'][plan['id_column']].array(library='np').tolist()}
        if op == 'weighted_hist':
            w = a[plan['weight']]
            return {'sumw': np.histogram(a[col], plan['edges'], weights=w)[0].tolist(),
                    'sumw2': np.histogram(a[col], plan['edges'], weights=w*w)[0].tolist()}
        with uproot.create(output_dir / 'hist.root') as out:
            out['hist'] = np.histogram(a[col], plan['edges'])
        with uproot.open(output_dir / 'hist.root') as out:
            values, edges = out['hist'].to_numpy()
            return {'values': values.tolist(), 'edges': edges.tolist()}


def cases(index, folder):
    """Each fixture has distinct schema/data. Plain-list oracles do not call executor."""
    tree, idc, scalar, jets, count, weight = [f'{x}_{index}' for x in ('Records', 'id', 'energy', 'objects_pt', 'nobjects', 'weight')]
    ids = [index*100 + i for i in range(6)]
    # Includes exact edges, flow, empty collections and negative weights.
    x = [-5., 0., 20.+index, 40., 80., 110.]
    w = [1., -0.5, 2., -1., 3., 0.]
    j = [[], [0., 30.], [35.+index], [10., 40., 60.], [], [5.]]
    n = [len(a) for a in j]
    if index % 2:
        n[2] += 1  # Deliberately malformed count branch for diagnosis.
    path = folder / f'fixture_{index}.root'
    with uproot.create(path) as f:
        t = f.mktree(tree, {idc: 'int64', scalar: 'float64', jets: 'var * float64', count: 'int32', weight: 'float64'})
        t.extend({idc: np.array(ids), scalar: np.array(x), jets: ak.Array(j), count: np.array(n, dtype='int32'), weight: np.array(w)})
        f['provenance'] = f'initial-{index}'
        f['provenance'] = f'revised-{index}'
    cut, edges = 20.+index, [0., 30., 60., 100.]
    selected = [eid for eid, v in zip(ids, x) if v > cut]
    sumw = [sum(wi for v, wi in zip(x, w) if lo <= v < hi) for lo, hi in zip(edges, edges[1:])]
    sumw2 = [sum(wi*wi for v, wi in zip(x, w) if lo <= v < hi) for lo, hi in zip(edges, edges[1:])]
    counts = [sum(lo <= v < hi for v in x) for lo, hi in zip(edges, edges[1:])]
    base = dict(tree=tree, column=scalar)
    yield_sum = sum(wi for v, wi in zip(x, w) if v > cut)
    definitions = [
        ('inventory', 'class', dict(object=tree), 'Inspect the class of the named event object.', 'TTree'),
        ('inventory', 'cycles', dict(object='provenance'), 'Read both saved provenance versions in chronological cycle order.', [f'initial-{index}', f'revised-{index}']),
        ('selection', 'select_gt', dict(**base, threshold=cut, id_column=idc), f'Find IDs with energy strictly greater than {cut} GeV.', selected),
        ('selection', 'select_range', dict(**base, low=0., high=40., id_column=idc), 'Find IDs with 0 <= energy < 40 GeV.', [eid for eid,v in zip(ids,x) if 0 <= v < 40]),
        ('jagged', 'jagged_count_gt', dict(tree=tree, column=jets, threshold=30.), 'Count objects above 30 GeV separately per row, including empty rows.', [sum(v > 30 for v in a) for a in j]),
        ('jagged', 'jagged_sum', dict(tree=tree, column=jets), 'Compute scalar object-pT sum per row in GeV.', [sum(a) for a in j]),
        ('histograms', 'weighted_hist', dict(**base, weight=weight, edges=edges), f'Compute signed sumw and sumw2 in bins {edges}; exclude flow and assume independent MC entries.', dict(sumw=sumw, sumw2=sumw2)),
        ('histograms', 'selected_yield', dict(**base, threshold=cut, weight=weight), f'Count selected rows and sum signed weights for energy > {cut} GeV.', dict(entries=len(selected), sumw=yield_sum)),
        ('validation_streaming', 'stream_sum', dict(**base, threshold=cut, weight=weight, chunk_entries=2), f'Accumulate signed weights for energy > {cut} GeV with exactly two rows per read chunk.', yield_sum),
        ('validation_streaming', 'check_counts', dict(tree=tree, column=jets, count=count, id_column=idc), 'Find IDs whose declared object count disagrees with collection length.', [eid for eid, a, c in zip(ids,j,n) if len(a) != c]),
        ('writing', 'write_skim', dict(**base, threshold=cut, id_column=idc), f'Write and reopen a new Selected TTree containing IDs and energy for energy > {cut} GeV. Preserve the input.', dict(class_='TTree', ids=selected)),
        ('writing', 'write_hist', dict(**base, edges=edges), f'Write and reopen an unweighted energy histogram with edges {edges}. Preserve the input.', dict(values=counts, edges=edges)),
    ]
    definitions[10][4]['class'] = definitions[10][4].pop('class_')
    context = (f'The harness supplies one local ROOT file. Its event object is {tree}; '
               f'ID column {idc}:int64, energy column {scalar}:float64 [GeV], '
               f'object pT collection {jets}:var * float64 [GeV], declared count {count}:int32, '
               f'signed MC weight {weight}:float64. String provenance has saved cycles. '
               'Actual values must be read from the file. All outputs use new harness-controlled paths.\n'
               'Available operations and required JSON arguments (plus operation):\n' + CONTRACT)
    for number, (category, op, params, question, expected) in enumerate(definitions):
        plan = dict(operation=op, **params)
        yield dict(id=f'filetask_{index}_{number:02d}', category=category, split_group=f'fixture_{index}',
                   fixture=path.name, prompt=context+'\n\nTask: '+question, reference_plan=plan, expected=expected)


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows))


def build(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    counts = {}
    for split, indices in [('train', [1, 2, 3]), ('validation', [7]), ('test', [11])]:
        folder = output / split
        folder.mkdir()
        rows = [r for i in indices for r in cases(i, folder)]
        for row in rows:
            before = digest(folder / row['fixture'])
            with tempfile.TemporaryDirectory(prefix='root-plan-') as tmp:
                actual = execute(row['reference_plan'], folder / row['fixture'], Path(tmp))
            if actual != row['expected'] or digest(folder / row['fixture']) != before:
                raise AssertionError(row['id'])
        write_jsonl(folder/'references.jsonl', rows)
        write_jsonl(folder/'prompts.jsonl', [dict(id=r['id'], prompt=r['prompt']) for r in rows])
        sft = [dict(id=r['id'], category=r['category'], split=split, split_group=r['split_group'], tools='[]',
                    messages=[dict(role='system', content=SYSTEM), dict(role='user', content=r['prompt']),
                              dict(role='assistant', content=json.dumps(r['reference_plan'], sort_keys=True))]) for r in rows]
        write_jsonl(folder/'sft.jsonl', sft)
        pq.write_table(pa.Table.from_pylist(sft), folder/'sft.parquet')
        counts[split] = dict(Counter(r['category'] for r in rows))
    hashes = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(dict(counts=counts, files_sha256=hashes,
        reference_checks_passed=60, distinct_operations=12, fixture_groups=5,
        system_prompt=SYSTEM, versions=dict(uproot=uproot.__version__, awkward=ak.__version__, numpy=np.__version__),
        limitations=['Template-related tasks across independent fixture groups; not 60 independent skills.',
                     'Same operation families across splits; measures schema/argument transfer, not unseen algorithms.',
                     'Single-operation plans with provided schemas, not open-ended code or autonomous exploration.']), indent=2)+'\n')


def evaluate(release, split, predictions, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((release/'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or digest(release/name) != expected:
            raise ValueError('Release integrity failure')
    refs = [json.loads(x) for x in (release/split/'references.jsonl').read_text().splitlines()]
    rows = [json.loads(x) for x in predictions.read_text().splitlines()]
    rows = [r for r in rows if r.get('record_type') != 'metadata']
    by_id = {r['id']: r for r in rows}
    if len(rows) != len(by_id) or set(by_id) != {r['id'] for r in refs}:
        raise ValueError('Missing, duplicate or unexpected prediction IDs')
    results = []
    for ref in refs:
        pred = by_id[ref['id']]
        if pred.get('prompt') != ref['prompt']:
            raise ValueError('Prompt mismatch')
        result = dict(id=ref['id'], category=ref['category'], passed=False)
        fixture = release/split/ref['fixture']
        before = digest(fixture)
        try:
            plan = json.loads(pred['output'])
            with tempfile.TemporaryDirectory(prefix='root-eval-') as tmp:
                actual = execute(plan, fixture, Path(tmp))
            # Require requested operation/arguments too: accidental equal results on a tiny file are insufficient.
            result.update(operation_match=plan == ref['reference_plan'], result_match=actual == ref['expected'])
            result['passed'] = result['operation_match'] and result['result_match'] and digest(fixture) == before
        except (ValueError, KeyError, TypeError, OSError, AttributeError) as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        results.append(result)
    cats = defaultdict(list)
    for row in results:
        cats[row['category']].append(row['passed'])
    output.mkdir(parents=True)
    write_jsonl(output/'results.jsonl', results)
    summary = dict(n=len(results), passed=sum(r['passed'] for r in results),
                   categories={k: dict(n=len(v), passed=sum(v), pass_rate=sum(v)/len(v)) for k,v in cats.items()},
                   prediction_sha256=digest(predictions), manifest_sha256=digest(release/'manifest.json'),
                   note='Allowlisted-operation execution and exact task-argument compliance; not arbitrary code execution accuracy.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('build')
    a.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('evaluate')
    b.add_argument('--release', type=Path, required=True)
    b.add_argument('--split', choices=['validation', 'test'], default='validation')
    b.add_argument('--predictions', type=Path, required=True)
    b.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.output)
        print('Verified 60 task instances; 36 train, 12 validation, 12 test.')
    else:
        print(json.dumps(evaluate(args.release, args.split, args.predictions, args.output), indent=2))
