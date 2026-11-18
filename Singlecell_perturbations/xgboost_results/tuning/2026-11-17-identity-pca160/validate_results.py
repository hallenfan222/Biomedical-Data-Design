import json,math,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
cs=json.loads((R/'configs.json').read_text());assert len(cs)==15
vs=[json.loads((R/'results'/(c['id']+'.json')).read_text()) for c in cs]
base=vs[0]['params'];seen=set()
for c,v in zip(cs,vs):
 assert v['id']==c['id'] and v['status']=='complete'
 assert v['params']=={**base,**c.get('overrides',{})}
 signature=json.dumps(v['params'],sort_keys=True);assert signature not in seen;seen.add(signature)
 assert [f['fold'] for f in v['folds']]==['NK','CD4','CD8','Treg']
 assert abs(v['mean_mrrmse']-sum(f['mrrmse'] for f in v['folds'])/4)<1e-12
 for f in v['folds']:
  n=127 if f['fold']=='CD8' else 129
  assert f['n_validation']==n and f['n_train']==614-n and f['prediction_shape']==[n,18211]
  assert f['device'].startswith('cuda') and f['peak_process_gpu_mib']>0
  assert math.isfinite(f['mrrmse']) and f['mrrmse']>=0
 for key in ['data_sha256','runner_sha256','split_sha256','xgboost','numpy']:
  assert v['context'][key]==vs[0]['context'][key]
smoke=json.loads((R/'results/T00.smoke.json').read_text())
assert abs(smoke['folds'][0]['mrrmse']-vs[0]['folds'][0]['mrrmse'])<1e-10
summary={'status':'PASS','unique_configs':len(vs),'complete_folds':sum(len(v['folds']) for v in vs),'same_source_and_data':True,'all_cuda':True,'smoke_matches_baseline_NK':True,'jobs':sorted(set(v['context']['job_id'] for v in vs))}
(R/'reports/validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

