import subprocess,json,time,sys,os,statistics
from pathlib import Path
R=Path.cwd();O=R.parent/'xgb-tuning-20261117';folds=['NK','CD4','CD8','Treg'];start=time.time();budget=5.8*3600
journal=R/'decisions.jsonl'
def note(**k):
 with journal.open('a') as h:h.write(json.dumps(dict(time=time.time(),**k))+'\n')
 print(k,flush=True)
def group(lib,d,tag,over,limit=600,maxproject=2400):
 rows=[]
 for f in folds:
  dst=R/'records'/f'{lib}-{d}-{tag}-{f}.json'; left=budget-(time.time()-start)
  if not dst.exists():
   if left<limit+300:note(skip=[lib,d,tag,f],reason='allocation time reserve');break
   t=time.time()
   try:subprocess.run([sys.executable,'-u','fit.py',lib,d,tag,f,json.dumps(over)],check=True,timeout=limit)
   except (subprocess.SubprocessError,OSError) as e:note(failed=[lib,d,tag,f],reason=str(e),elapsed=time.time()-t);break
  r=json.loads(dst.read_text());rows.append(r)
  if f=='NK' and (r['total_seconds']*4>maxproject or r['total_seconds']*3>left-300):note(skip=[lib,d,tag],reason='NK measured projected cost exceeds stage allowance',nk=r['total_seconds']);break
 if len(rows)==4:return dict(lib=lib,design=d,stage=tag,overrides=over,scores=[r['mrrmse'] for r in rows],seconds=sum(r['seconds'] for r in rows),mean=statistics.mean(r['mrrmse'] for r in rows))
def prior(t):
 v=json.loads((O/'results'/f'{t}.json').read_text());p=v['params'];return dict(lib='xgb',design='B',stage=t,overrides={k:p[k] for k in ['max_depth','learning_rate','n_estimators','reg_lambda']},scores=[f['mrrmse'] for f in v['folds']],seconds=v['model_total_seconds'],mean=v['mean_mrrmse'])
base=[prior('T00')];pool=[prior('T09')]
for lib,d,timeout in [('xgb','D',360),('pyboost','B',240),('pyboost','D',360),('xgb','A',1300),('xgb','C',900),('pyboost','A',750),('pyboost','C',750)]:
 if lib=='pyboost' and d in 'ABD' and Path('reuse.json').exists():
  historical=json.loads(Path('reuse.json').read_text());base.append(historical[d]);note(reuse=d,lib=lib);continue
 z=group(lib,d,'base',{},timeout,3000 if d in 'AC' else 2000)
 if z:base.append(z)
for lib in ['xgb','pyboost']:
 choices=sorted([v for v in base if v['lib']==lib],key=lambda v:v['mean']);pool.extend(choices)
 for b in choices[:2]:
  d=b['design'];note(selected=d,lib=lib,reason='two lowest complete baseline means; inspect folds and cost',baseline=b)
  if lib=='xgb' and d=='B':continue
  one={'max_depth':4,'reg_lambda':.1} if lib=='xgb' else {'max_depth':4}
  two={**one,**({'learning_rate':.03,'n_estimators':300} if lib=='xgb' else {'lr':.04,'ntrees':180})}
  for tag,over in [('structure',one),('schedule',two)]:
   z=group(lib,d,tag,over,900 if d in 'AC' else 550,3200)
   if z:pool.append(z)
   else:break
bestx=min([v for v in pool if v['lib']=='xgb'],key=lambda v:v['mean']);z=group('xgb',bestx['design'],'strategy',{**bestx['overrides'],'multi_strategy':'multi_output_tree'},500,1000)
if z:pool.append(z)
winners={lib:min([v for v in pool if v['lib']==lib],key=lambda v:v['mean']) for lib in ['xgb','pyboost'] if any(v['lib']==lib for v in pool)}
Path('locked.json').write_text(json.dumps(winners,indent=2));note(locked=winners)
for lib,w in winners.items():
 if Path('id_map.csv').exists():subprocess.run([sys.executable,'fit.py',lib,w['design'],w['stage'],'final',json.dumps(w['overrides'])],check=True)

