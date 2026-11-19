"""Build tables for the comparison report from measured folds and the decision journal."""
import json
from pathlib import Path

import pandas as pd


def sections(root, results, fold_names, output):
    events = [json.loads(line) for line in (root/'decisions.jsonl').read_text().splitlines()] if (root/'decisions.jsonl').exists() else []
    attempts = set(results[['lib','design','stage']].itertuples(index=False, name=None))
    failures = []
    for event in events:
        if 'failed' in event:
            lib, design, stage, fold = event['failed']
            attempts.add((lib, design, stage))
            failures.append(dict(library=lib, design=design, stage=stage, fold=fold, seconds=event.get('elapsed'), error=event['reason']))
    done = set(results.loc[results.fits.eq(4), ['lib','design','stage']].itertuples(index=False,name=None))
    counts = [dict(library=lib,design=design,attempted=len([a for a in attempts if a[:2]==(lib,design)]),complete=len([a for a in done if a[:2]==(lib,design)])) for lib in ['xgb','pyboost'] for design in 'ABCD']
    counts = pd.DataFrame(counts);counts.to_csv(output/'parameter_counts.csv',index=False)
    parts = ['<h2>Number of parameter sets evaluated</h2><p>Counts include reused, complete and interrupted validation configurations. A single fold never qualifies for ranking.</p>',counts.to_html(index=False)]
    if failures:
        note = ('<p>Design C combined 22,939 inputs with 18,211 direct outputs. The XGBoost NK test reached its 900-second time limit. The PyBoost NK test raised CuPy OutOfMemoryError while requesting 21,054,101,504 more bytes with 21,313,750,528 bytes already allocated. The effects of RNA at full output dimension and of PCA with RNA therefore remain unresolved for any library lacking complete C results.</p>')
        parts.extend(['<h2>Incomplete attempts</h2>',pd.DataFrame(failures).to_html(index=False),note])
    locked = root/'locked.json'
    if locked.exists() and (root/'xgb-selection-final.json').exists():
        kaggle_file=root/'kaggle.json'
        kaggle=json.loads(kaggle_file.read_text()) if kaggle_file.exists() else {}
        winners=[]
        for lib,w in json.loads(locked.read_text()).items():
            remote=kaggle.get(lib,{})
            rec=root/'records'/f"{lib}-{w['design']}-{w['stage']}-final.json"
            actual=json.loads(rec.read_text()) if rec.exists() else {}
            winners.append(dict(library=lib,design=w['design'],trial=w['stage'],**dict(zip(fold_names,w['scores'])),equal_weight_mean=w['mean'],validation_seconds=w['seconds'],full_fit_seconds=actual.get('seconds'),parameters=json.dumps(actual.get('params',w['overrides']),sort_keys=True),submission_id=remote.get('id'),public_score=remote.get('public'),private_score=remote.get('private')))
        table=pd.DataFrame(winners);table.to_csv(output/'final_models.csv',index=False)
        parts[:0]=['<h2>Final model comparison</h2><p>Selection used complete validation folds before Kaggle submission. Public and private scores are shown only when returned by Kaggle.</p>',table.to_html(index=False,na_rep='Pending / unavailable')]
    return parts
