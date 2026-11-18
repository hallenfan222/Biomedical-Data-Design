"""Build self-contained HTML/SVG and full CSV from saved four-fold JSON results. No fitting."""
import json,csv,html,math,statistics,sys
from pathlib import Path
R=Path(__file__).resolve().parent
O=R/'reports';O.mkdir(exist_ok=True)
configs=json.loads((R/'configs.json').read_text()); trials=[];failed=[]
for c in configs:
    p=R/'results'/(c['id']+'.json')
    if not p.exists():failed.append((c['id'],'not run'));continue
    v=json.loads(p.read_text())
    if v.get('status')=='complete' and len(v.get('folds',[]))==4:trials.append(v)
    else:failed.append((c['id'],v.get('error',v.get('status','incomplete'))))
assert trials,'No complete four-fold result'
trials.sort(key=lambda v:v['id']); base=next(v for v in trials if v['id']=='T00'); ranked=sorted(trials,key=lambda v:v['mean_mrrmse']);best=ranked[0];recommended=best
if (R/'recommendation.json').exists():recommended=next(v for v in trials if v['id']==json.loads((R/'recommendation.json').read_text())['id'])
F=['NK','CD4','CD8','Treg'];bm={x['fold']:x['mrrmse'] for x in base['folds']}
rows=[];longrows=[]
for v in trials:
    row=dict(trial_id=v['id'],stage=v['stage'],**v['params'],pca_dim=160,mean_mrrmse=v['mean_mrrmse'],delta_vs_baseline=v['mean_mrrmse']-base['mean_mrrmse'],improvement_pct=100*(base['mean_mrrmse']-v['mean_mrrmse'])/base['mean_mrrmse'],model_seconds=v['model_total_seconds'],seconds_including_preprocessing=v['total_with_preprocess_seconds'],slurm_job_id=v['context']['job_id'],log_path=str(R/'logs'/(v['context']['job_id']+'.log')),status=v['status'])
    for f in v['folds']:
        row[f['fold']+'_mrrmse']=f['mrrmse'];row[f['fold']+'_seconds']=f['model_seconds'];row[f['fold']+'_peak_gpu_mib']=f['peak_process_gpu_mib'];longrows.append(dict(trial_id=v['id'],job_id=v['context']['job_id'],**f))
    rows.append(row)
def csvwrite(path,arr):
    with path.open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(arr[0]));w.writeheader();w.writerows(arr)
csvwrite(O/'results.csv',rows);csvwrite(O/'fold_results.csv',longrows)
with (O/'trial_status.csv').open('w',newline='') as fp:
    w=csv.writer(fp);w.writerow(['trial_id','status_or_error']);w.writerows([(v['id'],'complete') for v in trials]+failed)

contrast_rows=[]
lookup={v['id']:v for v in trials}
pairs=[('T00','T01'),('T00','T02'),('T00','T03'),('T00','T04'),('T00','T05'),('T00','T06'),('T00','T07'),('T02','T08'),('T05','T08'),('T08','T09'),('T08','T10'),('T08','T11'),('T08','T12'),('T09','T11')]
ref=min([v for v in trials if v['stage']!='sampling'],key=lambda v:v['mean_mrrmse'])
pairs += [(ref['id'],v['id']) for v in trials if v['stage']=='sampling']
for a,b in pairs:
 if a not in lookup or b not in lookup:continue
 av,bv=lookup[a],lookup[b];changes={k:[av['params'][k],bv['params'][k]] for k in av['params'] if av['params'][k]!=bv['params'][k]}
 row={'reference':a,'trial':b,'changed_parameters':json.dumps(changes),'mean_delta':bv['mean_mrrmse']-av['mean_mrrmse'],'seconds_delta':bv['model_total_seconds']-av['model_total_seconds']}
 row.update({f+'_delta':next(x['mrrmse'] for x in bv['folds'] if x['fold']==f)-next(x['mrrmse'] for x in av['folds'] if x['fold']==f) for f in F});contrast_rows.append(row)
if contrast_rows:csvwrite(O/'controlled_contrasts.csv',contrast_rows)

def esc(s):return html.escape(str(s))
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.grid':True,'grid.alpha':.18})
def svg_finish(fig):
    fig.tight_layout();s=io.StringIO();fig.savefig(s,format='svg',bbox_inches='tight');plt.close(fig);return s.getvalue()[s.getvalue().index('<svg'):]
def bar_chart():
    fig,ax=plt.subplots(figsize=(11,max(4,len(trials)*.38)))
    labels=[v['id']+(' baseline' if v['id']=='T00' else '')+(' lowest' if v['id']==best['id'] else '')+(' recommended' if v['id']==recommended['id'] else '') for v in trials]
    vals=[v['mean_mrrmse'] for v in trials];colors=['#64748b' if v['id']=='T00' else '#059669' if v['id']==recommended['id'] else '#2563eb' for v in trials]
    ax.barh(labels,vals,color=colors);ax.invert_yaxis();ax.axvline(base['mean_mrrmse'],color='#dc2626',ls='--',label='T00 baseline')
    spread=max(vals)-min(vals);pad=max(spread*.25,.004);ax.set_xlim(min(vals)-pad,max(vals)+pad*1.6)
    for i,v in enumerate(vals):ax.text(v+pad*.07,i,f'{v:.6f}',va='center',fontsize=9)
    ax.set_xlabel('Equal four-fold mean MRRMSE (lower is better; truncated axis)');ax.set_title('All completed trials - actual trial order');ax.legend(loc='best');return svg_finish(fig)
def contrasts():
    groups=[('max_depth',['T00','T01','T02']),('min_child_weight',['T00','T03','T04']),('reg_lambda',['T00','T05','T06','T07'])]
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,(param,ids) in zip(axes,groups):
        vs=sorted([v for v in trials if v['id'] in ids],key=lambda v:v['params'][param]);xs=[v['params'][param] for v in vs];ys=[v['mean_mrrmse'] for v in vs];ax.plot(xs,ys,'o-',color='#2563eb')
        if param=='reg_lambda':ax.set_xscale('log')
        ax.set_xticks(xs,[str(x) for x in xs]);ax.set_xlabel(param);ax.set_title('Other parameters fixed to T00',fontsize=10)
        for x,y,v in zip(xs,ys,vs):ax.annotate(v['id']+' '+f'{y:.5f}',(x,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
    axes[0].set_ylabel('Four-fold mean MRRMSE');fig.suptitle('Controlled single-parameter contrasts',y=1.04);return svg_finish(fig)
def fold_chart(v):
    ds=[next(f['mrrmse'] for f in v['folds'] if f['fold']==k)-bm[k] for k in F];fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(F,ds,color=['#059669' if d<0 else '#dc2626' for d in ds]);ax.axhline(0,color='#111827',lw=1);pad=max(max(abs(d) for d in ds)*.25,.002);ax.set_ylim(min(ds+[0])-pad,max(ds+[0])+pad)
    for i,d in enumerate(ds):ax.annotate(f'{d:+.6f}',(i,d),xytext=(0,-15 if d<0 else 5),textcoords='offset points',ha='center')
    ax.set_ylabel('MRRMSE difference (recommended - baseline)');ax.set_title(v['id']+' vs T00: negative = improvement');return svg_finish(fig)
def pair_chart():
    vs=[v for v in trials if v['id']=='T08' or v['stage']=='rate_rounds']
    if not vs:return ''
    fig,ax=plt.subplots(figsize=(11,4));xs=list(range(len(vs)));ys=[v['mean_mrrmse'] for v in vs];ax.scatter(xs,ys,s=65,color='#7c3aed')
    ax.set_xticks(xs,[v['id']+'\nlr='+str(v['params']['learning_rate'])+' / rounds='+str(v['params']['n_estimators']) for v in vs]);ax.set_xlim(-.5,len(vs)-.5)
    for i,y in enumerate(ys):ax.annotate(f'{y:.6f}',(i,y),xytext=(0,10),textcoords='offset points',ha='center')
    ax.margins(y=.35);ax.set_ylabel('Four-fold mean MRRMSE');ax.set_title('Learning rate / rounds pairs; remaining parameters fixed to T08');return svg_finish(fig)
def sampling_chart():
    sam=[v for v in trials if v['stage']=='sampling']
    if not sam:return ''
    ref=min([v for v in trials if v['stage']!='sampling'],key=lambda v:v['mean_mrrmse']);vs=[ref]+sam
    fig,ax=plt.subplots(figsize=(10,4));ys=[v['mean_mrrmse'] for v in vs];ax.scatter(range(len(vs)),ys,s=65,color='#0891b2')
    ax.set_xticks(range(len(vs)),[v['id']+'\nsub='+str(v['params']['subsample'])+' / col='+str(v['params']['colsample_bytree']) for v in vs]);ax.set_xlim(-.5,len(vs)-.5);ax.margins(y=.4)
    for i,y in enumerate(ys):ax.annotate(f'{y:.6f}',(i,y),xytext=(0,10),textcoords='offset points',ha='center')
    ax.set_title('Sampling contrasts; remaining parameters fixed to '+ref['id']);ax.set_ylabel('Four-fold mean MRRMSE');return svg_finish(fig)

def shortlist_chart():
    vs=ranked[:5]+([] if any(v['id']=='T00' for v in ranked[:5]) else [base])
    fig,ax=plt.subplots(figsize=(11,4.2));ys=list(range(len(vs)));xs=[v['mean_mrrmse'] for v in vs]
    ax.scatter(xs,ys,s=80,c=['#059669' if v['id']==recommended['id'] else '#64748b' if v['id']=='T00' else '#2563eb' for v in vs])
    ax.set_yticks(ys,[v['id']+(' baseline' if v['id']=='T00' else ' recommended' if v['id']==recommended['id'] else '') for v in vs]);ax.invert_yaxis();ax.axvline(base['mean_mrrmse'],ls='--',color='#dc2626')
    pad=max(max(xs)-min(xs),.005)*.25;ax.set_xlim(min(xs)-pad,max(xs)+pad*2)
    for x,y,v in zip(xs,ys,vs):ax.annotate(f'{x:.6f} / {v["model_total_seconds"]:.1f}s',(x,y),xytext=(8,7),textcoords='offset points',fontsize=9)
    ax.set_xlabel('Four-fold mean MRRMSE (lower is better)');ax.set_title('Top five candidates and baseline - zoomed score / model time');return svg_finish(fig)

charts={'all_trials.svg':bar_chart(),'controlled_parameters.svg':contrasts(),'fold_deltas.svg':fold_chart(recommended),'rate_rounds.svg':pair_chart(),'sampling.svg':sampling_chart(),'shortlist.svg':shortlist_chart()}
for name,s in charts.items():(O/name).write_text(s)
def table(arr,cols):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(k)+'</th>' for k in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(f'{r[k]:.6f}' if isinstance(r.get(k),float) else r.get(k,''))+'</td>' for k in cols)+'</tr>' for r in arr)+'</tbody></table></div>'
journal=(R/'SEARCH_LOG.md').read_text() if (R/'SEARCH_LOG.md').exists() else 'Search journal pending.'
delta=recommended['mean_mrrmse']-base['mean_mrrmse'];pct=-100*delta/base['mean_mrrmse'];deltas={k:next(f['mrrmse'] for f in recommended['folds'] if f['fold']==k)-bm[k] for k in F};n_improve=sum(d<0 for d in deltas.values());largest=min(deltas,key=deltas.get)
recommend_text=f'推荐 {recommended["id"]}；本轮最低分为 {best["id"]}。相对 baseline T00，平均 MRRMSE 变化 {delta:+.6f}（改善 {pct:.2f}%）。四折中 {n_improve}/4 折改善，最大改善发生在 {largest}。这些是本轮搜索在固定四折上的选择结果，不是独立测试集结论，也不是统计显著性证明。'
cols=['trial_id','stage','NK_mrrmse','CD4_mrrmse','CD8_mrrmse','Treg_mrrmse','mean_mrrmse','delta_vs_baseline','improvement_pct','model_seconds','slurm_job_id']
paramcols=['trial_id','n_estimators','learning_rate','max_depth','min_child_weight','reg_lambda','subsample','colsample_bytree','multi_strategy','random_state','max_bin','pca_dim','objective','tree_method','device','n_jobs']
topids={v['id'] for v in ranked[:3]}|{'T00'};toprows=sorted([r for r in rows if r['trial_id'] in topids],key=lambda r:r['mean_mrrmse'])
body='<h1>BDD · XGBoost 调参报告</h1><p>Expanse · identity input · PCA 160 · seed 42 · 四折 sparse-cell validation</p><div class="callout">'+esc(recommend_text)+'</div><p>每折先计算各验证组合跨 18,211 个基因的 row RMSE，再对组合求均值得到 MRRMSE。排名使用四折 MRRMSE 的等权平均。验证行数为 NK/CD4/Treg 各 129、CD8 为 127；不把全部行混合后按行数加权。</p><h2>1. Baseline 与前 3 组</h2>'+table(toprows,cols)+table(toprows,paramcols)+'<h2>2. 哪些配置改善？</h2>'+charts['all_trials.svg']+charts['shortlist.svg']+'<h2>3. 哪些参数有直接对照证据？</h2><p>下图仅展示单变量对照；曲线连接实际评估点，不代表未评估点。三个小图共享纵轴，数值标在每个点上；L2 正则横轴使用对数刻度。</p>'+charts['controlled_parameters.svg']+charts['rate_rounds.svg']+charts['sampling.svg']+'<p>学习率与轮数是联合配置比较，不把联合变化的收益归给其中一个参数。结构参数组合和后续采样设置的选取过程见下方日志。</p><h2>4. 提升是否分布在多数折？</h2>'+charts['fold_deltas.svg']+table(toprows,['trial_id','NK_mrrmse','CD4_mrrmse','CD8_mrrmse','Treg_mrrmse','model_seconds'])+'<h2>5. 实际搜索过程与选参依据</h2><pre>'+esc(journal)+'</pre><h3>最终选参结论</h3><pre>'+esc((R/'CONCLUSION.md').read_text() if (R/'CONCLUSION.md').exists() else 'Search still running.')+'</pre><h3>可核对的参数对照</h3>'+table(contrast_rows,['reference','trial','changed_parameters','mean_delta','seconds_delta'])+'<h2>6. 全部完成配置（实际运行顺序）</h2>'+table(rows,cols)+table(rows,paramcols)+'<h2>7. 失败或未完成配置</h2><pre>'+esc(json.dumps(failed,ensure_ascii=False,indent=2))+'</pre><h2>8. 复现、来源和限制</h2><p>没有保存逐 boosting 轮的原基因空间 MRRMSE，因此不提供学习曲线，不以 PCA 空间 RMSE 替代，也没有为作图额外训练。使用固定轮数，无 early stopping。所有配置的 PCA/encoder 只在各折训练集拟合。</p><p>model_seconds 为四折训练、预测、评分和监控开销；CSV 另含预处理耗时。采样显存峰值是每 0.5 秒读取本进程 nvidia-smi 的下界。不同作业/顺序存在缓存和负载差异，时间用于粗略比较，不是严格性能基准。</p><pre>'+esc((R/'PROVENANCE.md').read_text())+'</pre><h3>推荐配置复现命令</h3><pre>'+esc('cd '+str(R)+'\nsbatch run.sbatch --ids '+recommended['id']+' --out reproduction-'+recommended['id'])+'</pre><h3>日志路径</h3>'+table(rows,['trial_id','slurm_job_id','log_path'])+'<h3>配套文件</h3><p><a href="results.csv">完整结果 CSV</a> · <a href="fold_results.csv">逐折结果 CSV</a> · <a href="trial_status.csv">全部试验状态</a> · <a href="../generate_report.py">图表生成脚本</a></p>'
style='body{font-family:Arial,"Noto Sans CJK SC",sans-serif;max-width:1250px;margin:36px auto;padding:0 28px;color:#1e293b;line-height:1.65;background:#f8fafc}h1{font-size:34px}h2{margin-top:42px}svg{width:100%;background:white;border:1px solid #dbe2eb;border-radius:12px;margin:12px 0}.callout{padding:20px;background:#dcfce7;border-left:5px solid #059669}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:12px;margin:14px 0;width:100%}td,th{padding:8px;border:1px solid #dbe2eb;text-align:left;white-space:nowrap}th{background:#e2e8f0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px;border:1px solid #dbe2eb;font-size:13px}a{color:#1d4ed8}@media print{body{margin:0;background:white}.scroll{overflow:visible}table{font-size:8px}svg,table{break-inside:avoid}}'
(O/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BDD XGBoost tuning</title><style>'+style+'</style><body>'+body+'</body></html>')
write={"recommended":recommended['id'],"best":best['id'],"delta":delta,"improvement_pct":pct,"fold_deltas":deltas,"top3":[v['id'] for v in ranked[:3]]}
(O/'summary.json').write_text(json.dumps(write,indent=2));print(json.dumps(write,indent=2));print('REPORT_READY',O/'report.html')

(O/'BDD_XGBoost_Tuning_Report.html').write_text((O/'report.html').read_text())
