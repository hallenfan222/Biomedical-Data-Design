"""BDD fixed sparse-cell XGBoost search. No test/Kaggle labels used."""
import os, sys, json, time, hashlib, argparse, subprocess, threading, csv, traceback, fcntl
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from split_data import make_sparse_splits, apply_split, get_gene_columns
CELL_NAMES=['NK cells','T cells CD4+','T cells CD8+','T regulatory cells']
FOLD_IDS=['NK','CD4','CD8','Treg']
BASE=dict(n_estimators=120,max_depth=3,learning_rate=0.06,min_child_weight=1.0,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,objective='reg:squarederror',tree_method='hist',device='cuda',n_jobs=2,random_state=42,max_bin=256,multi_strategy='one_output_per_tree')
def write_json(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)); tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def metric(a,b):
    return float(np.sqrt(np.mean((np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))**2,axis=1)).mean())
class Monitor:
    def __init__(self): self.stop=threading.Event(); self.samples=[]
    def poll(self):
        while not self.stop.is_set():
            try:
                out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=4)
                vals=[float(row.split(',')[1].strip()) for row in out.splitlines() if row.split(',')[0].strip()==str(os.getpid())]
                if vals:self.samples.append((time.time(),sum(vals)))
            except Exception: pass
            self.stop.wait(0.5)
    def __enter__(self): self.t=threading.Thread(target=self.poll,daemon=True); self.t.start(); return self
    def __exit__(self,*a): self.stop.set(); self.t.join(5)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config-file',default='configs.json');parser.add_argument('--ids',nargs='+',required=True);parser.add_argument('--smoke',action='store_true');parser.add_argument('--out',default='results');args=parser.parse_args()
    out=ROOT/args.out;out.mkdir(exist_ok=True,parents=True)
    lock=(ROOT/'gpu-run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('SLURM_JOB_ID'),'Run inside a Slurm GPU allocation'
    assert xgb.build_info().get('USE_CUDA'), 'CUDA XGBoost build required'
    configs=json.loads((ROOT/args.config_file).read_text()); assert len(configs)<=15
    selected=[c for c in configs if c['id'] in args.ids]; assert len(selected)==len(args.ids)
    data_path=ROOT/'data/de_train.parquet'; data_hash=sha(data_path)
    assert data_hash=='b5fa2f763dd98b7662ec8819209cf97e68022e33a575df77d43d1153f7c5ac76'
    data=pd.read_parquet(data_path);genes=get_gene_columns(data)
    assert data.shape==(614,18216) and len(genes)==18211
    manifests=make_sparse_splits(data); write_json(ROOT/'fold_manifests.json',manifests)
    context=dict(job_id=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),data_sha256=data_hash,runner_sha256=sha(Path(__file__)),split_sha256=sha(ROOT/'source/split_data.py'),xgboost=xgb.__version__,numpy=np.__version__,gpu=subprocess.check_output(['nvidia-smi','-L'],text=True).strip(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write_json(out/('context-'+context['job_id']+'.json'),context)
    print('CONTEXT',json.dumps(context),flush=True)
    prepared={}
    for fid,cell in zip(FOLD_IDS,CELL_NAMES):
        if args.smoke and fid!='NK':continue
        t=time.perf_counter();train,val=apply_split(data,manifests[cell])
        assert len(val)==(127 if fid=='CD8' else 129)
        encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
        a=encoder.fit_transform(train[['cell_type','sm_name']].astype(str));b=encoder.transform(val[['cell_type','sm_name']].astype(str))
        reducer=PCA(n_components=160,svd_solver='randomized',random_state=42)
        y=reducer.fit_transform(train[genes].to_numpy(dtype=np.float32)).astype(np.float32)
        prepared[fid]=(np.ascontiguousarray(a),np.ascontiguousarray(b),y,val[genes].to_numpy(dtype=np.float32),reducer,time.perf_counter()-t)
        print('PREPARED',fid,a.shape,y.shape,'seconds',prepared[fid][-1],flush=True)
    # Each fold's transform uses only that fold's train data. Reuse identical transforms across trials.
    for c in selected:
        dest=out/(c['id']+('.smoke' if args.smoke else '')+'.json')
        if dest.exists():
            old=json.loads(dest.read_text())
            if old.get('status')=='complete':print('SKIP complete',c['id'],flush=True);continue
            raise RuntimeError('Existing incomplete result: '+str(dest)+'; preserve it and choose another --out')
        result=dict(id=c['id'],stage=c['stage'],rationale=c['rationale'],params={**BASE,**c.get('overrides',{})},pca_dim=160,context=context,status='running',folds=[])
        write_json(dest,result);begin=time.perf_counter()
        try:
            for fid,(a,b,y,yval,reducer,prep_seconds) in prepared.items():
                start=time.perf_counter()
                with Monitor() as mon:
                    model=xgb.XGBRegressor(**result['params']);tf=time.perf_counter();model.fit(a,y);fit_seconds=time.perf_counter()-tf
                    config=json.loads(model.get_booster().save_config());device=config['learner']['generic_param']['device'];assert device.startswith('cuda'),device
                    assert config['learner']['learner_train_param']['multi_strategy']=='one_output_per_tree'
                    # DMatrix prediction avoids CPU-input inplace_predict device mismatch warnings.
                    latent=model.get_booster().predict(xgb.DMatrix(b));assert latent.shape==(len(b),160)
                    pred=reducer.inverse_transform(latent);assert pred.shape==yval.shape and np.isfinite(pred).all()
                    score=metric(yval,pred)
                row=dict(fold=fid,n_train=len(a),n_validation=len(b),n_genes=len(genes),input_dim=a.shape[1],mrrmse=score,fit_seconds=fit_seconds,model_seconds=time.perf_counter()-start,preprocess_seconds=prep_seconds,device=device,prediction_shape=list(pred.shape),peak_process_gpu_mib=max([x[1] for x in mon.samples],default=None),gpu_samples=len(mon.samples))
                result['folds'].append(row);write_json(dest,result)
                with (out/(c['id']+'-'+fid+('-smoke' if args.smoke else '')+'-gpu.csv')).open('w') as f:
                    w=csv.writer(f);w.writerow(['unix_time','process_gpu_mib']);w.writerows(mon.samples)
                print('FOLD',c['id'],json.dumps(row),flush=True)
                del model,pred,latent
            result.update(status='complete',mean_mrrmse=float(np.mean([r['mrrmse'] for r in result['folds']])),model_total_seconds=time.perf_counter()-begin,total_with_preprocess_seconds=time.perf_counter()-begin+sum(r['preprocess_seconds'] for r in result['folds']))
            if not args.smoke:assert len(result['folds'])==4
            write_json(dest,result);print('COMPLETE',c['id'],result['mean_mrrmse'],result['model_total_seconds'],flush=True)
        except Exception as e:
            result.update(status='failed',error=repr(e),traceback=traceback.format_exc());write_json(dest,result);print(result['traceback'],flush=True);raise
    print('RUN_COMPLETE',args.ids,flush=True)
if __name__=='__main__':main()
