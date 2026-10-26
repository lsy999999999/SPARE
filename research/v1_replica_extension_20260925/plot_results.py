"""Reproducible publication figures from accepted, frozen result tables."""
from pathlib import Path
import csv
import hashlib
import json
import os
import time
os.environ.setdefault('MPLCONFIGDIR','/tmp/kv-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/v1_replica_extension_20260925_r1'
OLD=ROOT/'artifacts/v1_mechanism_models_replicas_20260925_r1'
OUT=DATA/'figures'
OUT.mkdir(exist_ok=True)
MODELS=['qwen7b','smol360m']
MODEL_NAMES={'qwen7b':'Qwen2.5-7B','smol360m':'SmolLM2-360M','qwen05b':'Qwen2.5-0.5B'}
CAPS=[768,2048,4096]
REPS=[1,2,4]
SHAPES=['steady','burst_stop']
SHAPE_NAMES={'steady':'Steady','burst_stop':'Burst-stop'}
LOAD_NAMES={'fixed_total':'Fixed total: 64 requests','per_replica':'Fixed per-replica: 64 requests each'}
ARMS=['arrival','old','srp','native','sarathi_common','past_future_common']
NAMES={'arrival':'basic+A','old':'basic','srp':'SRP','native':'Native','sarathi_common':'Sarathi*','past_future_common':'Past-Future*'}
COLORS=['#0072B2','#56B4E9','#E69F00','#777777','#009E73','#CC79A7']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white',
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True})

def read_csv(p):
 with p.open() as f:return list(csv.DictReader(f))

comparisons=read_csv(DATA/'analysis/comparisons.csv')
runs=read_csv(DATA/'analysis/runs.csv')
probes=read_csv(OLD/'analysis/probes.csv')
assert len(runs)==624 and len(comparisons)==260
assert len({r['task'] for r in runs})==16
C={(r['model'],int(r['capacity']),int(r['replicas']),r['case'].removeprefix('conversation_w0_'),r['load'],r['reference']):r for r in comparisons}
R={}
for r in runs:
 key=(r['model'],int(r['capacity']),int(r['replicas']),r['case'].removeprefix('conversation_w0_'),r['load'],r['arm'])
 R.setdefault(key,[]).append(r)
assert all(len(v)==2 for v in R.values())
for k,rs in R.items():
 assert sorted(int(r['repeat']) for r in rs)==[0,1]

def load_for(rep,load):return 'fixed_total' if rep==1 else load

def pair(model,cap,rep,shape,load,ref):return C.get((model,cap,rep,shape,load_for(rep,load),ref))

def values(model,cap,rep,shape,load,arm,metric):
 rs=R.get((model,cap,rep,shape,load_for(rep,load),arm),[])
 return [float(r[metric]) for r in sorted(rs,key=lambda r:int(r['repeat']))]

def mean(*args):
 xs=values(*args)
 return float(np.mean(xs)) if xs else np.nan

# Independently check that plotted relative gains use mean TTFTs, not mean percentages.
for key,r in C.items():
 model,cap,rep,shape,load,ref=key
 a=mean(model,cap,rep,shape,load,'arrival','mean_ttft_ms')
 b=mean(model,cap,rep,shape,load,ref,'mean_ttft_ms')
 assert np.isclose(100*(1-a/b),float(r['gain_pct']))

saved=[]
def save(fig,name):
 for ext in ('png','pdf','svg'):
  fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight',pad_inches=.16)
 saved.append(name)
 plt.close(fig)

def heat(ax,arr,xlabels,ylabels,title,norm):
 cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#E4E7EB')
 im=ax.imshow(np.ma.masked_invalid(arr),cmap=cmap,norm=norm,aspect='auto')
 ax.set_xticks(range(len(xlabels)),xlabels)
 ax.set_yticks(range(len(ylabels)),ylabels)
 ax.tick_params(axis='both',length=0,pad=7)
 ax.set_title(title,pad=13,fontweight='bold')
 ax.set_xticks(np.arange(-.5,arr.shape[1],1),minor=True)
 ax.set_yticks(np.arange(-.5,arr.shape[0],1),minor=True)
 ax.grid(which='minor',color='white',linewidth=1.4)
 ax.tick_params(which='minor',bottom=False,left=False)
 for i in range(arr.shape[0]):
  for j in range(arr.shape[1]):
   v=arr[i,j]
   label='N/A' if not np.isfinite(v) else f'{v:+.1f}%'
   dark=np.isfinite(v) and (norm(v)<.19 or norm(v)>.82)
   ax.text(j,i,label,ha='center',va='center',fontsize=9,color='white' if dark else '#202A35')
 return im

# 1/2: all scenarios and four external baselines. One shared, unclipped color scale.
norm=TwoSlopeNorm(vmin=-140,vcenter=0,vmax=100)
for number,load in [(1,'fixed_total'),(2,'per_replica')]:
 fig,axes=plt.subplots(1,2,figsize=(13.3,11.3))
 rowkeys=[(cap,rep,shape) for cap in CAPS for rep in REPS for shape in SHAPES]
 refs=['srp','native','sarathi_common','past_future_common']
 for ax,model in zip(axes,MODELS):
  arr=np.full((len(rowkeys),4),np.nan)
  for i,(cap,rep,shape) in enumerate(rowkeys):
   for j,ref in enumerate(refs):
    r=pair(model,cap,rep,shape,load,ref)
    if r:arr[i,j]=float(r['gain_pct'])
  labels=[f'{cap:,} / {rep}R / {SHAPE_NAMES[shape]}' for cap,rep,shape in rowkeys]
  im=heat(ax,arr,[NAMES[a] for a in refs],labels,MODEL_NAMES[model],norm)
  for y in (5.5,11.5):ax.axhline(y,color='#64748B',lw=2)
  ax.set_xlabel('Comparison baseline')
 axes[0].set_ylabel('KV blocks per replica / replica count / arrival shape')
 fig.suptitle('Full method vs. baselines: mean TTFT improvement',fontsize=17,fontweight='bold',y=.975)
 fig.text(.5,.937,LOAD_NAMES[load]+'  |  Positive = basic+A is faster',ha='center',fontsize=12)
 fig.subplots_adjust(left=.18,right=.93,bottom=.13,top=.89,wspace=.65)
 cax=fig.add_axes([.30,.077,.49,.018]);cb=fig.colorbar(im,cax=cax,orientation='horizontal')
 cb.set_label('100 × (1 − TTFT[basic+A] / TTFT[baseline]) (%)')
 fig.text(.02,.012,'624 accepted runs; 2 repeats per method/scenario. Grey N/A = interrupted 4-replica / 768-block group, excluded.\n* Paper-rule adaptations, not full original systems. KV is per replica; one reused length trace. Single-replica data are shared across load views.',fontsize=9)
 save(fig,f'{number:02d}_full_method_{load}')

# 3: A-only ablation, same-state core method baseline, explicitly distinct from figures 1/2.
fig,axes=plt.subplots(2,2,figsize=(12,8.8))
norm_a=TwoSlopeNorm(vmin=-50,vcenter=0,vmax=50)
for i,model in enumerate(MODELS):
 for j,load in enumerate(('fixed_total','per_replica')):
  rowkeys=[(rep,shape) for rep in REPS for shape in SHAPES]
  arr=np.full((6,3),np.nan)
  for y,(rep,shape) in enumerate(rowkeys):
   for x,cap in enumerate(CAPS):
    r=pair(model,cap,rep,shape,load,'old')
    if r:arr[y,x]=float(r['gain_pct'])
  im=heat(axes[i,j],arr,[f'{c:,}' for c in CAPS],[f'{r}R / {SHAPE_NAMES[s]}' for r,s in rowkeys],
    MODEL_NAMES[model]+' | '+('Fixed total' if j==0 else 'Per-replica load'),norm_a)
  axes[i,j].set_xlabel('KV blocks per replica')
fig.suptitle('Incremental effect of A: basic+A vs. basic',fontsize=17,fontweight='bold',y=.98)
fig.subplots_adjust(left=.14,right=.95,bottom=.18,top=.90,wspace=.45,hspace=.43)
cax=fig.add_axes([.30,.102,.45,.018]);fig.colorbar(im,cax=cax,orientation='horizontal').set_label('Mean TTFT improvement (%); positive = faster')
fig.text(.02,.014,'Same core algorithm; only the arrival term changes. N/A denotes excluded groups, not zero gain.\nTwo repeats; values are point estimates, not significance claims. At 4096 blocks, no saving actions occurred.',fontsize=9)
save(fig,'03_arrival_term_ablation')

# 4: Absolute mean TTFT with both individual repeats, four independent replicas.
fig,axes=plt.subplots(2,2,figsize=(13,8.7))
width=.12
for i,model in enumerate(MODELS):
 for j,shape in enumerate(SHAPES):
  ax=axes[i,j]
  for k,arm in enumerate(ARMS):
   x=np.arange(2)+(k-2.5)*width
   ys=[mean(model,cap,4,shape,'per_replica',arm,'mean_ttft_ms') for cap in (2048,4096)]
   ax.bar(x,ys,width=.105,label=NAMES[arm],color=COLORS[k],zorder=2)
   for xx,cap in zip(x,(2048,4096)):
    ys2=values(model,cap,4,shape,'per_replica',arm,'mean_ttft_ms')
    ax.scatter([xx-.016,xx+.016],ys2,s=12,c='#17212C',zorder=4,alpha=.75)
  ax.set_xticks([0,1],['2,048','4,096']);ax.set_xlabel('KV blocks per replica')
  ax.set_ylabel('Mean TTFT (ms)');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.18,zorder=0)
  ax.set_title(MODEL_NAMES[model]+' | '+SHAPE_NAMES[shape],fontweight='bold')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=6,loc='upper center',bbox_to_anchor=(.5,.93),frameon=False)
fig.suptitle('Four independent replicas: absolute first-token latency',fontsize=17,fontweight='bold',y=.985)
fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.83,hspace=.40,wspace=.22)
fig.text(.03,.025,'64 requests per replica (256 total); each replica has its own GPU and KV cache. Lower is better.\nBars = mean of 2 repeats; black points = individual repeats (not confidence intervals). Panel y-axis ranges differ.\n* Sarathi and Past-Future are rule adaptations. The excluded 768-block groups are omitted.',fontsize=9)
save(fig,'04_four_replicas_absolute_ttft')

# 5: Hardware scaling only. Hold the algorithm fixed (basic+A), vary replica count.
fig,axes=plt.subplots(2,4,figsize=(17,8.2))
capcolors=['#0072B2','#E69F00','#009E73']
for col,(model,load) in enumerate((m,l) for m in MODELS for l in ('fixed_total','per_replica')):
 for cap,color in zip(CAPS,capcolors):
  for shape,style,marker in [('steady','-','o'),('burst_stop','--','s')]:
   ttft=[mean(model,cap,r,shape,load,'arrival','mean_ttft_ms') for r in REPS]
   rate=[mean(model,cap,r,shape,load,'arrival','throughput_tokens_s') for r in REPS]
   ratio=np.asarray(rate)/rate[0]
   axes[0,col].plot(REPS,ttft,style,marker=marker,color=color,lw=1.8,ms=5)
   axes[1,col].plot(REPS,ratio,style,marker=marker,color=color,lw=1.8,ms=5)
 if load=='per_replica':axes[1,col].plot(REPS,REPS,':',color='#9CA3AF',lw=1.5,zorder=0)
 axes[0,col].set_title(MODEL_NAMES[model]+'\n'+('Fixed total' if load=='fixed_total' else 'Per-replica load'),fontweight='bold')
 axes[0,col].set_yscale('log');axes[1,col].set_ylim(0,4.4)
 for row in (0,1):
  axes[row,col].set_xticks(REPS);axes[row,col].set_xlabel('Independent replicas');axes[row,col].grid(alpha=.17)
  axes[row,col].set_xlim(.8,4.2)
 axes[0,col].set_ylabel('Mean TTFT (ms, log scale)')
 axes[1,col].set_ylabel('Throughput / single replica')
legend=[Line2D([0],[0],color=color,lw=2,label=f'{cap:,} KV blocks / replica') for cap,color in zip(CAPS,capcolors)]
legend += [Line2D([0],[0],color='#334155',ls=style,marker=marker,label=SHAPE_NAMES[shape]) for shape,style,marker in [('steady','-','o'),('burst_stop','--','s')]]
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.925),ncol=5,frameon=False)
fig.suptitle('Scaling with independent replicas: hold basic+A fixed',fontsize=17,fontweight='bold',y=.985)
fig.subplots_adjust(left=.065,right=.985,bottom=.15,top=.79,wspace=.41,hspace=.33)
fig.text(.03,.027,'Adding replicas also adds GPUs and total KV; these curves show hardware scaling, not the isolated contribution of A.\nFixed total = 64 requests globally. Per-replica load = 64 requests each. Dotted grey line = ideal throughput scaling (per-replica panels only).\n768-block / 4-replica points are missing because those groups were excluded. Each point averages two repeats.',fontsize=9)
save(fig,'05_replica_scaling')

# 6: Descriptive interventions and separate conditional single-action probes.
fig,(ax,bx)=plt.subplots(1,2,figsize=(13,6.8),gridspec_kw={'width_ratios':[1,1.25]})
for model,color in zip(MODELS,['#0072B2','#D55E00']):
 for shape,style,marker in [('steady','-','o'),('burst_stop','--','s')]:
  ys=[mean(model,cap,1,shape,'fixed_total','arrival','interventions') for cap in CAPS]
  ax.plot(range(3),ys,style,marker=marker,color=color,lw=2,ms=6,label=MODEL_NAMES[model]+' / '+SHAPE_NAMES[shape])
ax.set_xticks(range(3),['768','2,048','4,096']);ax.set_xlabel('KV blocks (one replica)')
ax.set_ylabel('Saving interventions per episode');ax.set_ylim(-15,760);ax.grid(axis='y',alpha=.18)
ax.set_title('A. Saving actions disappear at high capacity',fontweight='bold')
ax.legend(fontsize=8,frameon=False,loc='upper right')
ax.annotate('All 4096-block extension trajectories:\n0 saving actions; 0 A/basic action differences',xy=(2,0),xytext=(.35,310),
 arrowprops=dict(arrowstyle='->',color='#475569'),fontsize=9,color='#334155',
 bbox=dict(boxstyle='round,pad=.4',facecolor='white',edgecolor='#CBD5E1',alpha=.95))
ys=np.arange(len(probes));gains=[float(r['equivalent_mean_ttft_saved_ms']) for r in probes]
bx.barh(ys,gains,color=['#0072B2' if x>=0 else '#D55E00' for x in gains],height=.6,zorder=2)
for y,r in zip(ys,probes):
 vals=[float(r['gain_r0_area_ms'])/64,float(r['gain_r1_area_ms'])/64]
 bx.scatter(vals,[y-.10,y+.10],color='#17212C',s=20,zorder=3)
bx.set_yticks(ys,[MODEL_NAMES[r['model']]+f" / round {r['round']}\nsaved {r['saving_blocks']} blocks" for r in probes],fontsize=9)
bx.invert_yaxis();bx.axvline(0,color='#64748B',lw=1);bx.grid(axis='x',alpha=.18)
bx.set_xlabel('Equivalent mean TTFT saved (ms; positive = better)')
bx.set_title('B. Matched-state, one-action probes (768 blocks)',fontweight='bold')
fig.suptitle('Mechanism evidence: inactivity and state-dependent action value',fontsize=16,fontweight='bold',y=.985)
fig.subplots_adjust(left=.07,right=.985,bottom=.23,top=.88,wspace=.60)
fig.text(.02,.045,'A: extension series; two-repeat averages, descriptive action counts. B: original probe series; a different controlled protocol, not pooled.\nB changes one saving action vs. no saving, then continues basic+A in both branches. Bars average two repeats; dots show each repeat.\nWaiting-area difference is divided by all 64 requests. Selected states are conditional examples, not whole-policy average effects.\n4096-block probes had no eligible intervention states; no counterfactual branches were executed there.',fontsize=9)
save(fig,'06_mechanism_evidence')

source_files=[DATA/'analysis/comparisons.csv',DATA/'analysis/runs.csv',DATA/'analysis/scaling.csv',DATA/'analysis/summary.json',OLD/'analysis/probes.csv']
manifest=dict(created=time.time(),accepted_extension_runs=len(runs),accepted_extension_groups=16,
 completed_scenario_cells=len(C)//5,figures=saved,formats=['png','pdf','svg'],
 source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 scope='Original extension snapshot only; no supplemental results pooled. Probe data are separately identified.',
 validation='624 rows; 16 accepted groups; 2 repeats for every scenario/method; all relative gains recomputed from absolute TTFT.')
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ('accepted_extension_runs','completed_scenario_cells','figures')},indent=2))
