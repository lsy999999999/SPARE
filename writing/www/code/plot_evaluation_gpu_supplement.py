"""Verified GPU supplement: figures and paired statistics, no new experiment selection."""
from pathlib import Path
from collections import defaultdict
import csv,json,statistics as st,random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_evaluation_revision import save, OUT, COLORS
ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'artifacts/www_evaluation_gpu_completion_20261003_r1'
def read(p): return list(csv.DictReader(p.open()))
rows=read(DATA/'analysis/runs.csv'); design=json.loads((DATA/'design.json').read_text())
cells={c['key']:c for c in design['cells']}
by=defaultdict(list)
for r in rows: by[r['cell'],r['arm']].append(r)
agg={r['arm']:r for r in read(DATA/'analysis/aggregate.csv')}
rel={r['reference']:r for r in read(DATA/'analysis/relative_to_arrival.csv')}
def effect(cell,arm):
 return 100*(1-st.fmean(float(r['ttft_ms_mean']) for r in by[cell,'arrival'])/st.fmean(float(r['ttft_ms_mean']) for r in by[cell,arm]))
def ci(values):
 rng=random.Random(61006); xx=sorted(st.fmean(rng.choices(values,k=len(values))) for _ in range(10000));return xx[249],xx[9749]
# Common color semantics throughout all GPU panels.
labels={'arrival':'SPARE','native':'vLLM V1 default','sarathi':'Sarathi rule','past_future':'Past-Future rule','srp':'Same-order / no saving','fixed256':'Fixed budget 256','fixed512':'Fixed budget 512','constant16':'Fixed threshold 16','constant32':'Fixed threshold 32','basic':'Without arrival term','no_finish_guard':'Without finish guard','budget_only':'Budget only','chunk_only':'Chunk only','token_proxy':'Token-based saving'}
colors={**COLORS,'fixed256':'#c29342','fixed512':'#d9a06b','constant16':'#238a83','constant32':'#72aba3','no_finish_guard':'#c29342','budget_only':'#8173b1','chunk_only':'#c77051','token_proxy':'#238a83'}
# Grouped comparison: GPU-measured absolute service outcomes.
fig,axes=plt.subplots(2,3,figsize=(12,7.2),layout='constrained')
arms=['arrival','native','sarathi','past_future','srp']
specs=[('ttft_ms_mean',.001,'Mean TTFT (s) ↓'),('ttft_ms_p99',.001,'P99 TTFT (s) ↓'),('e2e_ms_mean',.001,'Mean E2E (s) ↓'),('output_throughput_tps',1,'Output tokens/s ↑'),('slo_ttft5000_maxitl200_goodput_rps',1,'SLO goodput (requests/s) ↑'),('recomputed_tokens',.001,'Recomputed tokens (thousands) ↓')]
for ax,(metric,scale,title) in zip(axes.flat,specs):
 vals=[float(agg[a][metric])*scale for a in arms]
 bars=ax.barh(range(5),vals,color=[colors[a] for a in arms],height=.65)
 ax.set_yticks(range(5),[labels[a] for a in arms],fontsize=9);ax.invert_yaxis();ax.set_xlabel(title);ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
 ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8);ax.set_xlim(0,max(vals)*1.2)
fig.suptitle('GPU supplement · 36 scenarios · four repeats · 128 requests per window')
save(fig,'E8_gpu_comparison')
# Capacity stratification is paired; source and arrival stay visible.
fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for ax,controls,title in zip(axes,[['srp','native','past_future'],['fixed256','constant16','fixed512'],['budget_only','chunk_only','token_proxy']],['Reference policies','Simple controls','Candidate and saving ablations']):
 for idx,a in enumerate(controls):
  values=[st.fmean(effect(c,a) for c in cells if cells[c]['capacity']==cap) for cap in [768,2048,4096]]
  ax.plot([768,2048,4096],values,color=colors[a],marker=['o','s','^'][idx],label=labels[a],lw=2,ms=5)
 ax.axhline(0,color='#7d8794',lw=.8,ls='--');ax.set_xticks([768,2048,4096]);ax.set_title(title);ax.set_xlabel('Configured KV blocks');ax.set_ylabel('SPARE mean TTFT reduction (%)');ax.grid(alpha=.15);ax.legend(fontsize=7.5,loc='best')
save(fig,'E9_gpu_capacity')
fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
for ax,controls,title in zip(axes,[['basic','no_finish_guard','token_proxy'],['budget_only','chunk_only','srp'],['constant16','constant32','fixed256','fixed512']],['State terms and safeguards','Candidate directions','Adaptive vs. fixed control']):
 for j,a in enumerate(controls):
  r=rel[a];m=float(r['arrival_mean_ttft_improvement_pct']);lo=float(r['bootstrap_window_95ci_low_pct']);hi=float(r['bootstrap_window_95ci_high_pct'])
  ax.barh(j,m,color=colors[a],height=.54)
  ax.errorbar(m,j,xerr=[[m-lo],[hi-m]],fmt='none',ecolor='#344259',capsize=3,lw=1.2)
  ax.text(.98,j,f'{m:+.2f}%',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
 ax.set_yticks(range(len(controls)),[labels[a] for a in controls],fontsize=9);ax.invert_yaxis();ax.axvline(0,color='#7d8794',lw=.8,ls='--');ax.set_title(title);ax.set_xlabel('SPARE mean TTFT reduction (%)');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('GPU ablations · bars: paired mean · error bars: 95% CI clustered by six source windows')
save(fig,'E10_gpu_ablations')
# Complete post-hoc threshold grid; the original primary threshold is unchanged.
thresholds=[(t,i) for t in [1000,2000,5000,10000] for i in [100,200,500]]
slo=defaultdict(list)
for r in rows:
 if r['arm'] not in arms: continue
 data=json.loads((DATA/r['result_path']).read_text())
 for t,i in thresholds:
  requests=data['requests']
  passed=sum(q['ttft_ms']<=t and (q['max_itl_ms'] is None or q['max_itl_ms']<=i) for q in requests)
  slo[r['arm'],t,i].append(passed/len(requests)*100)
fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for ax,a in zip(axes,['native','sarathi','past_future']):
 matrix=np.array([[st.fmean(slo['arrival',t,i])-st.fmean(slo[a,t,i]) for i in [100,200,500]] for t in [1000,2000,5000,10000]])
 im=ax.imshow(matrix,cmap='PuOr_r',vmin=-60,vmax=60,aspect='auto')
 for j in range(4):
  for k in range(3):ax.text(k,j,f'{matrix[j,k]:+.1f}',ha='center',va='center',fontsize=11,color='white' if abs(matrix[j,k])>40 else '#25344a')
 ax.set_xticks(range(3),[100,200,500]);ax.set_yticks(range(4),[1,2,5,10]);ax.set_xlabel('Maximum ITL threshold (ms)');ax.set_ylabel('TTFT threshold (s)');ax.set_title('SPARE − '+labels[a]+' (pp)')
fig.suptitle('SLO attainment difference · joint TTFT and maximum-ITL requirements')
save(fig,'E11_slo_grid')
summary={'paired_capacity':{str(cap):{a:st.fmean(effect(c,a) for c in cells if cells[c]['capacity']==cap) for a in rel} for cap in [768,2048,4096]},'slo_grid':[{ 'arm':a,'ttft_ms':t,'max_itl_ms':i,'attainment_percent':st.fmean(slo[a,t,i])} for a in arms for t,i in thresholds], 'scope':'2016 accepted runs, four repeats averaged, equal scenario weights'}
(OUT/'supplement_statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Wrote E8–E11 and supplement_statistics.json')
