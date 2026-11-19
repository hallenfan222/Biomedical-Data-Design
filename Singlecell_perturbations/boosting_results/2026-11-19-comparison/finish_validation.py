"""Use spare allocation time to finish the deferred raw-target baseline before final fitting."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

FOLDS = ('NK', 'CD4', 'CD8', 'Treg')

def record(**event):
    with Path('decisions.jsonl').open('a') as handle:
        print(json.dumps({'time': time.time(), **event}), file=handle, flush=True)

def resolve_xgb():
    selected = Path('locked.json')
    choices = json.loads(selected.read_text())
    if Path('xgb-selection-final.json').exists():
        return choices['xgb']
    def path(fold):
        return Path('records') / f'xgb-A-base-{fold}.json'
    measured = json.loads(path('NK').read_text())['total_seconds'] if path('NK').exists() else None
    missing = [f for f in FOLDS if not path(f).exists()]
    deadline = os.environ.get('SLURM_JOB_END_TIME') or os.environ.get('BDD_JOB_DEADLINE')
    left = max(0, int(deadline) - int(time.time())) if deadline else 0
    reserve = 3300
    bound = max(1350, int(1.6 * measured)) if measured else 1800
    if measured and left > len(missing) * bound + reserve:
        record(resume='xgb/A/base', remaining_seconds=left, per_fold_limit=bound, final_reserve=reserve, reason='PCA experiments and strategy comparison finished; the existing NK result leaves enough time for the other folds.')
        for fold in missing:
            try:
                subprocess.run([sys.executable, '-u', 'fit.py', 'xgb', 'A', 'base', fold, '{}'], timeout=bound, check=True)
            except (subprocess.SubprocessError, OSError) as exc:
                record(failed=['xgb','A','base',fold], reason=str(exc));break
    elif missing:
        record(deferred_remaining=missing, remaining_seconds=left, reason='Insufficient time after reserving both final fits')
    if all(path(f).exists() for f in FOLDS):
        rows = [json.loads(path(f).read_text()) for f in FOLDS]
        scores = [r['mrrmse'] for r in rows]
        candidate = dict(lib='xgb', design='A', stage='base', overrides={}, scores=scores, mean=statistics.mean(scores), seconds=sum(r['seconds'] for r in rows))
        if candidate['mean'] < choices['xgb']['mean']:
            choices['xgb'] = candidate
    temp = selected.with_suffix('.tmp')
    temp.write_text(json.dumps(choices, indent=2));temp.replace(selected)
    Path('xgb-selection-final.json').write_text(json.dumps(choices['xgb'], indent=2))
    record(final_selection=choices, reason='Complete equal-weight four-fold scores before either final prediction or Kaggle submission')
    return choices['xgb']
