"""Export reviewed experiment artifacts to a ZIP and final predictions to the user's Portal home."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORTAL = Path.home() / 'BDD_BOOSTING_COMPARISON_20261119'
SCRIPTS = ['fit.py', 'campaign.py', 'batch.sh', 'resume.sbatch', 'finish_validation.py', 'validate_submission.py', 'build_report.py', 'report_tables.py', 'paired_results.py', 'export_results.py']
RESULTS = ['reuse.json', 'decisions.jsonl', 'locked.json', 'xgb-selection-final.json']


def export():
    PORTAL.mkdir(exist_ok=True)
    files = [ROOT / name for name in SCRIPTS + RESULTS]
    files += sorted((ROOT/'records').glob('*.json'))
    files += [p for p in (ROOT/'report').rglob('*') if p.is_file() and p.suffix in {'.csv','.html','.png'}]
    files += [ROOT/name for name in ('accounting.txt','kaggle.json','validation.json','README.md','environment.txt','log-excerpts.txt','experiment_notes.md') if (ROOT/name).is_file()]
    package = PORTAL / 'BDD_BOOSTING_COMPARISON_20261119.zip'
    for lib in ('xgb','pyboost'):
        final = ROOT / f'{lib}-final.csv'
        if not final.is_file() or not list((ROOT/'records').glob(f'{lib}-*-final.json')):
            raise RuntimeError(f'Missing final fit or CSV for {lib}')
    with zipfile.ZipFile(package.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, str(p.relative_to(ROOT)))
    package.with_suffix('.tmp').replace(package)
    for name in ('xgb-final.csv','pyboost-final.csv','report/comparison.html','report/trials.csv','report/folds.csv'):
        target = PORTAL/Path(name).name
        shutil.copyfile(ROOT/name,target.with_suffix('.tmp'))
        target.with_suffix('.tmp').replace(target)
    print(json.dumps({'zip':str(package),'files':len(files),'sha256':hashlib.sha256(package.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':
    export()
