"""Build the comparison report from saved fits, with no model training."""
from pathlib import Path
import base64, csv, html, io, json, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
previous = root.parent / 'xgb-tuning-20261117'
out = root / 'report'
out.mkdir(exist_ok=True)
folds = ['NK','CD4','CD8','Treg']
rows = []
def load(path):
    return json.loads(path.read_text())
for trial, stage in [('T00','base'),('T09','T09')]:
    source = previous/'results'/f'{trial}.json'
    v = load(source if source.exists() else root/'provenance'/f'{trial}.json')
    for f in v['folds']:
        rows.append(dict(lib='xgb',design='B',stage=stage,fold=f['fold'],mrrmse=f['mrrmse'],seconds=f['model_seconds'],peak=f['peak_process_gpu_mib'],job=v['context']['job_id'],source='Expanse archive '+trial,params=json.dumps(v['params'],sort_keys=True)))
for d,v in load(root/'reuse.json').items():
    for cell,score in zip(folds,v['scores']):
        rows.append(dict(lib='pyboost',design=d,stage='base',fold=cell,mrrmse=score,seconds=None,peak=None,job=None,source='Colab run_e927a290c474',params='historical PyBoost baseline'))
for path in sorted((root/'records').glob('*.json')):
    v=load(path)
    if v['fold'] not in folds:continue
    rows.append({**v,'source':'Expanse comparison','params':json.dumps(v['params'],sort_keys=True)})
df=pd.DataFrame(rows); keys=['lib','design','stage'];assert not df.duplicated(keys+['fold']).any();df.to_csv(out/'folds.csv',index=False)
s=df.groupby(keys).agg(fits=('fold','count'),mean=('mrrmse','mean'),seconds=('seconds',lambda x:x.sum(min_count=1)),peak_mib=('peak','max'),source=('source','first'),params=('params','first')).reset_index()
s.loc[s.fits.ne(4),'mean']=np.nan
for d,v in load(root/'reuse.json').items():s.loc[(s.lib=='pyboost')&(s.design==d)&(s.stage=='base'),'seconds']=v['seconds']
s.to_csv(out/'trials.csv',index=False); complete=s[s.fits==4].copy(); complete['label']=complete.lib+'/'+complete.design+'/'+complete.stage
body=['<h1>BDD boosting comparison</h1>','<p>Selection uses the equal-weight mean of four gene-space MRRMSEs. A: identity, B: identity + output PCA160, C: identity + baseline RNA, D: identity + baseline RNA + output PCA160. Lower scores are better. Colab and Expanse timings come from different hardware and include different overheads.</p>']
def figure(title):
    fig,ax=plt.subplots(figsize=(9,4.5));ax.set_title(title);return fig,ax
def save(fig,name):
    fig.tight_layout();b=io.BytesIO();fig.savefig(b,format='png',dpi=140,bbox_inches='tight');plt.close(fig);(out/(name+'.png')).write_bytes(b.getvalue());body.append('<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">')
b=complete[complete.stage=='base'];p=b.pivot(index='design',columns='lib',values='mean').reindex(list('ABCD'));fig,ax=figure('Baseline comparison · missing bars have no complete four-fold result');p.plot.bar(ax=ax,rot=0);ax.set_ylabel('Mean MRRMSE');save(fig,'baseline')
for name, contrasts in [('RNA',[('A','C'),('B','D')]),('PCA',[('A','B'),('C','D')])]:
    fig,ax=figure(name+' effect with baseline parameters · negative favors the added feature or transform');plotted=0
    for lib in ['xgb','pyboost']:
        v=df[(df.lib==lib)&(df.stage=='base')].pivot(index='fold',columns='design',values='mrrmse').reindex(folds)
        for a,c in contrasts:
            if a in v and c in v and v[[a,c]].notna().all().all():ax.plot(folds,v[c]-v[a],marker='o',label=f'{lib}: {c} minus {a}');plotted+=1
    ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Paired fold MRRMSE difference');
    if plotted:ax.legend()
    save(fig,name.lower()+'-effect')
v=df.pivot(index=keys,columns='fold',values='mrrmse').reindex(columns=folds);fig,ax=plt.subplots(figsize=(7,max(4,.4*len(v))));im=ax.imshow(v,aspect='auto',cmap='viridis_r');ax.set_xticks(range(4),folds);ax.set_yticks(range(len(v)),['/'.join(k) for k in v.index]);fig.colorbar(im,ax=ax,label='MRRMSE');
for i in range(len(v)):
    for j in range(4):
        if pd.notna(v.iloc[i,j]):ax.text(j,i,f'{v.iloc[i,j]:.3f}',ha='center',va='center',bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=.4))
save(fig,'fold-scores')
fig,ax=figure('Time and validation score · historical runs identified separately');
for _,v in complete.iterrows():
    ax.scatter(v.seconds,v['mean']);ax.annotate(v.label,(v.seconds,v['mean']),fontsize=7,rotation=18)
ax.set_xscale('log');ax.set_xlabel('Four-fold recorded seconds (log scale)');ax.set_ylabel('Mean MRRMSE');save(fig,'runtime')
for lib in ['xgb','pyboost']:
    sub=complete[(complete.lib==lib)&(complete.stage!='strategy')]; tuned=sub[sub.stage!='base']; fig,ax=figure(f'{lib} tuning · complete four-fold candidates');
    for _,v in tuned.iterrows():
        base=sub[(sub.design==v.design)&(sub.stage=='base')];
        if len(base):ax.plot(['Baseline','Candidate'],[base.iloc[0]['mean'],v['mean']],marker='o',label=v.label)
    if len(tuned):ax.legend(fontsize=7)
    save(fig,lib+'-tuning')
strategy=complete[complete.stage=='strategy'];
for _,v in strategy.iterrows():
    params=json.loads(v.params);params['multi_strategy']='one_output_per_tree'; parent=complete[(complete.lib=='xgb')&(complete.design==v.design)&(complete.params==json.dumps(params,sort_keys=True))];
    if len(parent):
        fig,ax=figure('XGBoost strategy comparison · identical resolved parameters except strategy');ax.bar(['one_output_per_tree','multi_output_tree'],[parent.iloc[0]['mean'],v['mean']]);ax.set_ylabel('Mean MRRMSE');save(fig,'tree-strategy')
body+=['<h2>Configuration counts and recorded results</h2>',s.groupby(['lib','design']).agg(attempts_with_a_saved_fold=('stage','count'),complete=('fits',lambda x:sum(x==4))).to_html(),s.to_html(index=False,na_rep='Incomplete / unavailable'),'<h2>Decision history and incomplete attempts</h2>']
for f in ['decisions.jsonl','locked.json','accounting.txt','kaggle.json']:
    path=root/f
    if path.exists():body+=['<h3>'+f+'</h3><pre>'+html.escape(path.read_text())+'</pre>']
body+=['<h2>Historical linear reference</h2><p>PCA + MRRMSE + L2: four-fold MRRMSE 0.969221, Kaggle private score 0.858.</p>']
from paired_results import paired_results
for paragraph in (root/'experiment_notes.md').read_text().strip().split(chr(10)*2):
    if paragraph.startswith('## '): body.append('<h2>'+html.escape(paragraph[3:])+'</h2>')
    else: body.extend('<p>'+html.escape(line)+'</p>' for line in paragraph.splitlines())
body.extend(paired_results(df, folds, out))
from report_tables import sections
body[2:2] = sections(root, s, folds, out)
(out/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>BDD boosting comparison</title><style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:24px}img{max-width:100%}pre,td{overflow-wrap:anywhere;white-space:pre-wrap}table{border-collapse:collapse;font-size:13px}td,th{padding:5px}</style>'+''.join(body))
