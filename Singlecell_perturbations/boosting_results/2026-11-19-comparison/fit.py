import argparse,json,os,sys,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
O=R.parent/'xgb-tuning-20261117'
sys.path.insert(0,str(O))
import runner as old
from runner import np,pd,xgb,OneHotEncoder,PCA
import cupy as cp
from py_boost import GradientBoosting
lib,design,stage,fold=sys.argv[1:5]
over=json.loads(sys.argv[5]) if len(sys.argv)>5 else {}
if fold=='final' and lib=='xgb' and (R/'locked.json').exists():
 from finish_validation import resolve_xgb
 winner=resolve_xgb();design,stage,over=winner['design'],winner['stage'],winner['overrides']
out=R/'records'/f'{lib}-{design}-{stage}-{fold}.json'
assert os.environ['SLURM_JOB_ID'] and cp.cuda.runtime.getDeviceCount()==1
if out.exists():
 assert json.loads(out.read_text())['overrides']==over
 sys.exit(0)
path=O/'data/de_train.parquet'; assert old.sha(path)=='b5fa2f763dd98b7662ec8819209cf97e68022e33a575df77d43d1153f7c5ac76'
d=pd.read_parquet(path);g=old.get_gene_columns(d); assert d.shape==(614,18216)
m=old.make_sparse_splits(d);assert m==json.loads((O/'fold_manifests.json').read_text())
if fold=='final':
 tr=d;va=pd.read_csv(R/'id_map.csv');assert va.id.is_unique and len(va)==255
else: tr,va=old.apply_split(d,m[old.CELL_NAMES[old.FOLD_IDS.index(fold)]])
t=time.monotonic(); enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32); cols=['cell_type','sm_name']; a=enc.fit_transform(tr[cols]);b=enc.transform(va[cols]);y=tr[g].to_numpy(dtype=np.float32)
if design in 'CD':
 path=O/'source/repo/Singlecell_perturbations/xgboost_results/baseline_expression_donor_balanced_v2.csv';assert old.sha(path)=='65d070bbf441e80f5b5354f59f191fb94992e377fa57635e5653488f0b6619e1'
 p=pd.read_csv(path,index_col=0);a=np.column_stack([a,p.loc[tr.cell_type].to_numpy(dtype=np.float32)]);b=np.column_stack([b,p.loc[va.cell_type].to_numpy(dtype=np.float32)])
if design in 'BD':
 pc=PCA(n_components=160,svd_solver='randomized',random_state=42);y=pc.fit_transform(y).astype(np.float32)
params=dict(old.BASE) if lib=='xgb' else dict(loss='mse',ntrees=120,lr=.06,max_depth=3,min_data_in_leaf=1,subsample=.9,colsample=.9,lambda_l2=1.,es=0,seed=42,verbose=10)
params.update(over);start=time.monotonic()
with old.Monitor() as mon:
 if lib=='xgb':
  mod=xgb.XGBRegressor(**params).fit(a,y).get_booster();cfg=json.loads(mod.save_config())['learner'];assert cfg['generic_param']['device'].startswith('cuda') and cfg['learner_train_param']['multi_strategy']==params['multi_strategy'];pred=mod.predict(xgb.DMatrix(b))
 else:
  mod=GradientBoosting(**params);mod.fit(a,y);pred=cp.asnumpy(mod.predict(b))
 if design in 'BD':pred=pc.inverse_transform(pred)
 assert pred.shape==(len(va),len(g)) and np.isfinite(pred).all()
 score=None if fold=='final' else old.metric(va[g].to_numpy(dtype=np.float32),pred)
 if fold=='final':
  df=pd.DataFrame(pred,columns=g);df.insert(0,'id',va.id);df.to_csv(R/f'{lib}-final.csv',index=False)
r=dict(lib=lib,design=design,stage=stage,fold=fold,params=params,overrides=over,mrrmse=score,seconds=time.monotonic()-start,total_seconds=time.monotonic()-t,peak=max([v for _,v in mon.samples],default=None),n_input=a.shape[1],n_output=y.shape[1],job=os.environ['SLURM_JOB_ID']);old.write_json(out,r);print(r,flush=True)
