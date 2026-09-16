"""Write a portable before/after report and training curve for a pipeline run."""
import csv
import json
from pathlib import Path
import re
import sys


def main():
    out = Path(sys.argv[1]).resolve()
    analysis = out / 'analysis'
    analysis.mkdir(exist_ok=True)
    summary = json.loads((out / 'pipeline-summary.json').read_text())
    split = summary.get('eval_split', 'test')
    model_name = ''
    protocol_path = out / 'base/protocol.json'
    if protocol_path.exists():
        model_name = json.loads(protocol_path.read_text()).get('base_model', '')
    lines = [f'# ROOT-command {model_name or "model"} results', '',
             f'Both models answer the same 560 {split} prompts with the same decoding protocol.', '',
             '| Model | Passed | Total | Success rate |', '|---|---:|---:|---:|']
    scores = {}
    for model in ('base', 'lora'):
        path = analysis / f'{model}-score/{model}-report.json'
        if path.exists():
            report = json.loads(path.read_text())
            scores[model] = report['models'][model]
            s = scores[model]
            lines.append(f"| {model} | {s['passed']} | {report['total']} | {s['accuracy']:.2%} |")
        else:
            lines.append(f'| {model} | pending | 560 | ROOT scoring not run |')
    log = out / '02-train-export.log'
    recovery = out / 'recovery-source.json'
    if recovery.exists():
        source = json.loads(recovery.read_text())['checkpoint']
        repo = Path(__file__).resolve().parents[1]
        checkpoint = repo / source.removeprefix('/workspace/')
        log = checkpoint.parent.parent / 'training.log'
        lines += ['', f'Recovered checkpoint: `{source}`. No training was performed in this run.',
                  'The configuration file records launcher settings; consult the source training log for actual recovered-model hyperparameters.']
    if log.exists():
        text = log.read_text()
        losses = [(int(s), float(v)) for s, v in re.findall(r'step:(\d+) - .*?train/loss:([\d.eE+-]+)', text)]
        validation = [(int(s), float(v)) for s, v in re.findall(r'step:(\d+) - val/loss:([\d.eE+-]+)', text)]
        with (analysis / 'training-loss.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'training_loss'])
            writer.writerows(losses)
        if losses:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(*zip(*losses), alpha=.3, label='Training loss')
            smooth = [sum(v for _, v in losses[max(0, i-49):i+1]) / min(i+1, 50) for i in range(len(losses))]
            ax.plot([s for s, _ in losses], smooth, label='50-step mean')
            for i, (step, _) in enumerate(validation):
                ax.axvline(step, color='red', linestyle='--', alpha=.6, label='Epoch boundary' if i == 0 else None)
            if validation:
                ax.plot(*zip(*validation), 'o', color='black', label='Validation loss')
            ax.set(xlabel='Training step', ylabel='Loss', yscale='log')
            ax.legend()
            fig.tight_layout()
            for suffix in ('png', 'pdf'):
                fig.savefig(analysis / f'training-loss.{suffix}')
            plt.close(fig)
            lines += ['', '![Training loss](analysis/training-loss.png)']
    lines += ['', '## Artifacts', '',
              '- `configuration.txt`: launcher parameters; `training-settings.json`: training settings.',
              '- `base/` and `lora/`: prompts, decoding protocol, completions, and inference timing.',
              '- `analysis/base-score/` and `analysis/lora-score/`: ROOT execution records and JSON/CSV reports, including results by API family.',
              '- `checkpoints/`: saved training states (recovery runs reference the original checkpoint).',
              '- `job.log`, stage logs/status JSON, `score-*.log`, and `status.txt`: execution and errors.',
              '', 'Use validation questions to select changes; reserve test scores for reporting.']
    (out / 'RESULTS_REPORT.md').write_text('\n'.join(lines) + '\n')
    summary = json.loads((out / 'pipeline-summary.json').read_text())
    summary['root_execution_scoring'] = 'completed' if len(scores) == 2 else 'not_run'
    summary['scores'] = scores
    (out / 'pipeline-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('\n'.join(lines[:8]))
    print(f'Report: {out / "RESULTS_REPORT.md"}')


if __name__ == '__main__':
    main()
