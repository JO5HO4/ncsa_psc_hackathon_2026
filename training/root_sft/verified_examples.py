#!/usr/bin/env python3
"""Build a balanced, executable training-candidate pack from synthetic ROOT files.

Only executes the trusted snippets defined below, never model-generated code.
The original question bank and evaluation splits are not modified.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import awkward as ak
import numpy as np
import uproot

SOURCE = 'https://uproot.readthedocs.io/en/latest/basic.html'
CONTEXT = ('Synthetic fixture.root contains a TTree Events with four rows in this order: '
           'event:int64=[101,102,103,104], met:float64=[20,40,80,10] GeV, '
           'weight:float64=[1,-0.5,2,3], Jet_pt:var * float64=[[10,40],[],[35],[20,60,5]] GeV. '
           'The scalar nJet:int32=[2,0,1,3] counts jets. Metadata is a TObjString written '
           'twice as note;1="first" and note;2="second". Use Python with Uproot 5, '
           'Awkward 2 and NumPy. Store the requested JSON-compatible output in result.')

# Each pair exercises different operations, not paraphrases of one answer.
CASES = [
    ('inventory', 'Which class stores Events? Inspect the file rather than inferring it from its suffix.',
     "with uproot.open(path) as f:\n    result = f.classnames(cycle=False)['Events']", 'TTree',
     'A ROOT file is a container; inspect object classes before selecting a reader.'),
    ('inventory', 'Retrieve both stored note cycles explicitly, oldest then newest.',
     "with uproot.open(path) as f:\n    result = [str(f['note;1']), str(f['note;2'])]", ['first', 'second'],
     'Key cycles distinguish stored versions of the same named object, not loop iterations.'),
    ('selection', 'Return event IDs with met strictly greater than 30 GeV.',
     "with uproot.open(path) as f:\n    a = f['Events'].arrays(['event', 'met'], library='np')\n    result = a['event'][a['met'] > 30].tolist()", [102, 103],
     'The event mask must select the event-ID array in the same row order.'),
    ('selection', 'Return event IDs satisfying both nJet >= 2 and met < 30 GeV.',
     "with uproot.open(path) as f:\n    a = f['Events'].arrays(['event', 'met', 'nJet'], library='np')\n    result = a['event'][(a['nJet'] >= 2) & (a['met'] < 30)].tolist()", [101, 104],
     'Use elementwise boolean operators and parentheses for array cuts.'),
    ('jagged', 'Count jets with pT strictly above 30 GeV separately for every event, retaining empty events.',
     "with uproot.open(path) as f:\n    pt = f['Events']['Jet_pt'].array(library='ak')\n    result = ak.to_list(ak.sum(pt > 30, axis=1))", [1, 0, 1, 1],
     'Reduce along the jet axis; flattening first would lose event boundaries.'),
    ('jagged', 'Compute scalar summed jet pT per event in GeV, with zero for an event containing no jets.',
     "with uproot.open(path) as f:\n    pt = f['Events']['Jet_pt'].array(library='ak')\n    result = ak.to_list(ak.sum(pt, axis=1))", [50.0, 0.0, 35.0, 85.0],
     'This is a scalar pT sum, not the magnitude of a vector sum.'),
    ('weighted_histograms', 'For met bins [0,30,60,100] GeV, compute sumw and sumw2. Keep signed weights; assume independent MC entries.',
     "with uproot.open(path) as f:\n    a = f['Events'].arrays(['met', 'weight'], library='np')\n    edges = [0, 30, 60, 100]\n    sumw = np.histogram(a['met'], bins=edges, weights=a['weight'])[0]\n    sumw2 = np.histogram(a['met'], bins=edges, weights=a['weight'] ** 2)[0]\n    result = {'sumw': sumw.tolist(), 'sumw2': sumw2.tolist()}",
     {'sumw': [4.0, -0.5, 2.0], 'sumw2': [10.0, 0.25, 4.0]},
     'Signed bin content is not an event count. For independent entries, the statistical uncertainty estimate is sqrt(sumw2); do not discard negative weights.'),
    ('weighted_histograms', 'For met > 30 GeV report selected row count and signed sum of weights separately.',
     "with uproot.open(path) as f:\n    a = f['Events'].arrays(['met', 'weight'], library='np')\n    mask = a['met'] > 30\n    result = {'entries': int(mask.sum()), 'sumw': float(a['weight'][mask].sum())}",
     {'entries': 2, 'sumw': 1.5},
     'Two selected entries need not represent a weighted yield of two.'),
    ('streaming_validation', 'Read only met and weight in chunks of two entries and accumulate sum of weights for met > 30 GeV.',
     "total = 0.0\nwith uproot.open(path) as f:\n    for a in f['Events'].iterate(['met', 'weight'], step_size=2, library='np'):\n        total += float(a['weight'][a['met'] > 30].sum())\nresult = total", 1.5,
     'Accumulate across all chunks; do not reset the total inside the loop. Integer step_size counts entries.'),
    ('streaming_validation', 'Validate nJet against each event\'s actual Jet_pt length; return the IDs of mismatching events.',
     "with uproot.open(path) as f:\n    a = f['Events'].arrays(['event', 'nJet', 'Jet_pt'], library='ak')\n    mismatch = a['nJet'] != ak.num(a['Jet_pt'], axis=1)\n    result = ak.to_list(a['event'][mismatch])", [],
     'Compare per-event lengths, not merely the number of rows in each branch.'),
    ('writing', 'Write a new TTree Selected containing only event and met for met > 30 GeV, reopen it and return its class and IDs. Do not overwrite the input.',
     "with uproot.open(path) as f:\n    a = f['Events'].arrays(['event', 'met'], library='np')\n    mask = a['met'] > 30\n    selected = {k: v[mask] for k, v in a.items()}\nwith uproot.create(out_path) as f:\n    tree = f.mktree('Selected', {'event': 'int64', 'met': 'float64'})\n    tree.extend(selected)\nwith uproot.open(out_path) as f:\n    result = {'class': f['Selected'].classname, 'event': f['Selected']['event'].array(library='np').tolist()}",
     {'class': 'TTree', 'event': [102, 103]},
     'Explicit mktree requests a TTree. uproot.create refuses an existing output path.'),
    ('writing', 'Write an unweighted met histogram with edges [0,30,60,100] GeV to a new ROOT file, reopen it and return bin contents and edges.',
     "with uproot.open(path) as f:\n    met = f['Events']['met'].array(library='np')\nwith uproot.create(out_path) as f:\n    f['met_hist'] = np.histogram(met, bins=[0, 30, 60, 100])\nwith uproot.open(out_path) as f:\n    values, edges = f['met_hist'].to_numpy()\n    result = {'values': values.tolist(), 'edges': edges.tolist()}",
     {'values': [2.0, 1.0, 1.0], 'edges': [0.0, 30.0, 60.0, 100.0]},
     'This histogram is deliberately unweighted. Preserve its bin edges when reading it back.'),
]


def fixture(path):
    with uproot.create(path) as f:
        tree = f.mktree('Events', {'event': 'int64', 'met': 'float64', 'weight': 'float64',
                                 'nJet': 'int32', 'Jet_pt': 'var * float64'})
        tree.extend({'event': np.array([101, 102, 103, 104]), 'met': np.array([20., 40., 80., 10.]),
                     'weight': np.array([1., -0.5, 2., 3.]), 'nJet': np.array([2, 0, 1, 3], dtype='int32'),
                     'Jet_pt': ak.Array([[10., 40.], [], [35.], [20., 60., 5.]])})
        f['note'] = 'first'
        f['note'] = 'second'


def build(output):
    if output.exists():
        raise FileExistsError(output)
    rows = []
    with tempfile.TemporaryDirectory(prefix='root-verified-') as tmp:
        path = Path(tmp) / 'fixture.root'
        fixture(path)
        for i, (kind, question, code, expected, explanation) in enumerate(CASES, 1):
            env = dict(uproot=uproot, ak=ak, np=np, path=path, out_path=Path(tmp) / f'output-{i}.root')
            exec(compile(code, f'trusted-reference-{i}', 'exec'), env)
            actual = env['result']
            if actual != expected:
                raise AssertionError((i, actual, expected))
            executable = ('import uproot\nimport awkward as ak\nimport numpy as np\n'
                          "path = 'fixture.root'\nout_path = 'answer.root'  # must not exist\n" + code)
            answer = f'{explanation}\n\n```python\n{executable}\n```\n\nExpected result: {json.dumps(expected)}'
            rows.append(dict(id=f'root_verified_{i:03d}', task_type=kind, question=question, context=CONTEXT,
                             answer=answer, reference_code=executable, expected_result=expected,
                             verification='executed_reference_passed', split_group='synthetic_events_fixture_v1',
                             intended_use='training_candidate_only', review_status='execution_verified_not_expert_certified',
                             provenance=[SOURCE], messages=[{'role': 'user', 'content': CONTEXT + '\n\n' + question},
                                                            {'role': 'assistant', 'content': answer}]))
    output.mkdir(parents=True)
    data = ''.join(json.dumps(row) + '\n' for row in rows)
    (output / 'examples.jsonl').write_text(data)
    (output / 'verification.json').write_text(json.dumps(dict(
        passed=len(rows), counts=dict(Counter(r['task_type'] for r in rows)),
        sha256=hashlib.sha256(data.encode()).hexdigest(),
        versions=dict(uproot=uproot.__version__, awkward=ak.__version__, numpy=np.__version__),
        warning='Synthetic training candidates, not held-out evaluation. All share one fixture and must stay in one split.'), indent=2) + '\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(f'Verified {len(build(args.output))} examples')
