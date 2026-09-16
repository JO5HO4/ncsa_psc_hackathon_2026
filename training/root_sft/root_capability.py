#!/usr/bin/env python3
"""Freeze ten file-operation prompts, or generate answers with the original Qwen.

Model-generated Python is saved as text and NEVER executed by this runner.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
PACK = REPO/'data/root_io/capability-v1'
MODEL = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
REVISION = '2e1fd397ee46e1388853d2af2c993145b0f1098a'
SYSTEM = 'You are a careful ROOT and high-energy-physics tutor. Answer the question clearly. State relevant assumptions and do not invent file contents, units, or analysis results.'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()


def prepare(sample):
    import uproot
    import awkward as ak
    import numpy as np
    if (PACK/'manifest.json').exists():raise FileExistsError('Pack already frozen; do not overwrite')
    if not sample.is_file():raise FileNotFoundError(sample)
    with uproot.open(sample) as f:
        classes=f.classnames(recursive=True,cycle=True)
        trees={k:dict(entries=f[k].num_entries,types=f[k].typenames())
               for k,c in f.classnames(recursive=True,cycle=False).items() if c=='TTree'}
    fixture=PACK/'fixture.root'
    incompatible=PACK/'incompatible.root'
    # Synthetic teaching data are explicitly separate from the real ROOT file.
    schema={'event':'int64','value':'float64','nItem':'int32',
            'Item_x':'var * float64','Item_y':'var * float64'}
    with uproot.create(fixture) as f:
        f.mktree('Events',schema).extend(dict(event=np.array([101,102,102,104],dtype='int64'),
            value=np.array([0,30,31,np.nan],dtype='float64'),nItem=np.array([2,0,1,2],dtype='int32'),
            Item_x=ak.Array([[1.,2.],[],[3.],[4.]]),Item_y=ak.Array([[5.,6.],[],[7.],[8.,9.]])))
        f['h_values']=(np.array([2.,3.,1.]),np.array([0.,1.,2.,3.]))
    with uproot.create(incompatible) as f:
        f.mktree('Events',{'event':'int64','value':'int32'}).extend(
            dict(event=np.array([201],dtype='int64'),value=np.array([4],dtype='int32')))
    metadata=dict(sample_path=str(sample.relative_to(REPO)),sample_sha256=sha(sample),
        sample_objects=classes,sample_trees=trees,
        supplement='fixture.root is synthetic, NOT content observed in the real sample.',
        fixture_types=schema,fixture_histogram='h_values: TH1D, three visible bins',
        incompatible='incompatible.root has Events with event:int64 and value:int32; fixture.root has value:float64')
    write(PACK/'sample_metadata.json',metadata)
    # Supply schemas, not reference implementations or expected numeric answers.
    context='Reference metadata from a read-only inspection:\n'+json.dumps(metadata,indent=2)
    common=('Return a runnable Python function and a short explanation. Use Uproot 5, Awkward 2 and NumPy '
            'unless the question explicitly requests PyROOT (ROOT 6). No tools are available in this conversation. '
            'Do not claim to have executed code or report invented file results. File and branch names are '
            'function arguments, not constants. Treat files as trusted benchmark inputs.\n\n'+context+'\n\nTask:\n')
    rows=[json.loads(l) for l in (PACK/'questions.jsonl').read_text().splitlines()]
    if len(rows)!=10 or len({r['id'] for r in rows})!=10:raise ValueError('Expected ten unique tasks')
    (PACK/'prompts.jsonl').write_text(''.join(json.dumps(dict(id=r['id'],prompt=common+r['question']))+'\n' for r in rows))
    write(PACK/'manifest.json',dict(model=MODEL,revision=REVISION,system_prompt=SYSTEM,
        temperature=0,max_new_tokens=1024,n=10,purpose='Pre-training code-generation diagnostic; not training data',
        files_sha256={n:sha(PACK/n) for n in ['questions.jsonl','prompts.jsonl','sample_metadata.json','fixture.root','incompatible.root']},
        preparation_packages={p:version(p) for p in ['uproot','awkward','numpy']},
        limitations=['No tool access or execution of generated answers.','Questions differ from the earlier conceptual 50-question test.',
                     'Reference code and correctness grading are not yet implemented for this pack.',
                     'The same 1024-token cap may truncate a longer function.']))
    print('Prepared ten tasks:',PACK)


def run(directory):
    manifest=read(PACK/'manifest.json')
    for name,expected in manifest['files_sha256'].items():
        if sha(PACK/name)!=expected:raise ValueError(f'Changed benchmark input: {name}')
    if (manifest['model'],manifest['revision'],manifest['system_prompt'],manifest['temperature'],manifest['max_new_tokens']) != (MODEL,REVISION,SYSTEM,0,1024):
        raise ValueError('Frozen configuration differs from runner')
    if directory.exists():raise FileExistsError(f'Refusing to overwrite {directory}; use a new RUN_DIR')
    import torch
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required')
    directory.mkdir(parents=True)
    started=now()
    write(directory/'status.json',dict(status='running',started_utc=started))
    try:
        path=snapshot_download(repo_id=MODEL,revision=REVISION,
            allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
        if (Path(path)/'lora_adapter').exists():raise ValueError('Expected original model without a project adapter')
        write(directory/'runtime.json',dict(model=MODEL,revision=REVISION,resolved_path=path,
            gpu=torch.cuda.get_device_name(0),packages={p:version(p) for p in ['torch','transformers','huggingface-hub']},
            manifest_sha256=sha(PACK/'manifest.json'),runner_sha256=sha(Path(__file__)),
            inference_sha256={p:sha(REPO/p) for p in ['inference/run_prompts.py','inference/model_runtime.py']},
            settings=manifest))
        with (directory/'inference.log').open('w') as log:
            subprocess.run([sys.executable,'-u','inference/run_prompts.py','--model',path,
                '--prompts',str(PACK/'prompts.jsonl'),'--output',str(directory/'answers.jsonl'),
                '--format','chat','--device','cuda','--temperature','0','--max-new-tokens','1024',
                '--system-prompt',SYSTEM],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        raw=[json.loads(l) for l in (directory/'answers.jsonl').read_text().splitlines()]
        rows=[r for r in raw if r['record_type']=='completion']
        prompts={r['id']:r['prompt'] for r in map(json.loads,(PACK/'prompts.jsonl').read_text().splitlines())}
        if len(rows)!=10 or {r['id'] for r in rows}!=set(prompts):raise ValueError('Missing or duplicate answers')
        if any(r['prompt']!=prompts[r['id']] or not r['output'].strip() for r in rows):raise ValueError('Wrong prompt or empty answer')
        report=['# Qwen ROOT file handling: ten-question baseline','',
            'Original Qwen2.5-Coder-1.5B-Instruct. Generated code has NOT been executed or correctness-graded.','']
        questions={r['id']:r for r in map(json.loads,(PACK/'questions.jsonl').read_text().splitlines())}
        review=[]
        for r in rows:
            q=questions[r['id']]
            report += [f"## {r['id']} - {q['category']}",'',q['question'],'','### Generated answer','',r['output'],'']
            review.append(dict(q,model_answer=r['output'],correctness=None,critical_error=None))
        (directory/'ANSWERS.md').write_text('\n'.join(report))
        write(directory/'review.json',review)
        write(directory/'status.json',dict(status='complete',started_utc=started,ended_utc=now(),answers=10,
            answers_sha256=sha(directory/'answers.jsonl')))
        print('Completed ten answers. Review:',directory/'ANSWERS.md',flush=True)
    except Exception as e:
        write(directory/'status.json',dict(status='failed',started_utc=started,ended_utc=now(),error=str(e)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--sample',type=Path,default=REPO/'data/samples/examples/Ntuple/tt_aMCNloP8.root')
    p.add_argument('--directory',type=Path)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.sample.resolve())
    elif a.directory:run(a.directory.resolve())
    else:p.error('run requires --directory')
