"""Time-weighted KV occupancy, variability and queue evidence."""
from __future__ import annotations
import csv
import gzip
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/www_kv_utilization_20261015_r1'
FIG = ROOT/'writing/www/figures/newidea'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

METHODS = (
 ('arrival','SPARE','#195c9b','-','o'),
 ('native','vLLM V1','#494949','--','s'),
 ('sarathi','Sarathi rule','#439bc2','-.','^'),
 ('sola_rule','SOLA rule','#d25a35',':','v'),
 ('uniboost_static','UniBoost rule','#9562a0','--','D'),
 ('past_future','Past-Future rule','#238a83','-.','P'),
 ('s3_proxy','S3 proxy','#b64342',':','X'),
 ('apt_serve_kv_rule','Apt-Serve rule','#a67b1e','--','*'))
LABELS={a:b for a,b,*_ in METHODS}
CAPS=(768,1024,2048,4096)
WARM, END = 60000.,120000.

def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def integrate(rows, arrivals, left, right, capacity):
    area=area2=waiting=high=low_wait=0.
    covered=0.
    for r in rows:
        a,b=r['simulated_start_ms'],r['simulated_finish_ms']
        lo,hi=max(a,left),min(b,right)
        if hi<=lo: continue
        dt=hi-lo
        u=(capacity-1-r['free_after'])/(capacity-1)
        assert 0<=u<=1
        n=r['pending_first_tokens']
        area+=u*dt;area2+=u*u*dt;covered+=dt
        high+=(u>=.95)*dt
        # Submitted requests are pending for the complete step; arrivals
        # during the step start waiting at their actual arrival timestamp.
        new=arrivals[(arrivals>a)&(arrivals<hi)]
        extra=float(np.maximum(0,hi-np.maximum(lo,new)).sum())
        waiting+=n*dt+extra
        if u<.8:
            if n>0: low_wait+=dt
            elif len(new): low_wait+=hi-max(lo,float(new[0]))
    duration=right-left
    assert covered<=duration+1e-6
    mean=area/duration
    return dict(mean_kv_pct=100*mean,sd_kv_pp=100*np.sqrt(max(0,area2/duration-mean*mean)),
        waiting_first_mean=waiting/duration, near_full_pct=100*high/duration,
        waiting_with_headroom_pct=100*low_wait/duration)

def assemble():
    design=json.loads((OUT/'design.json').read_text())
    summaries=[]; series=[]; hashes={}
    for s in design['samples']:
        for c in CAPS:
            for arm,*_ in METHODS:
                path=OUT/'runs'/f"{s['name']}_c{c}_{arm}"
                result=json.loads((path/'result.json').read_text())
                hashes.setdefault(s['name'],set()).add(result['request_sha256'])
                with gzip.open(path/'iterations.jsonl.gz','rt') as f: rows=[json.loads(x) for x in f]
                assert len(rows)==result['rounds']
                assert all(a['simulated_finish_ms']<=b['simulated_start_ms']+1e-7 for a,b in zip(rows,rows[1:]))
                arrivals=np.array([r['arrival_ms'] for r in json.loads((path/'input.json').read_text())])
                meta=dict(source=s['source'],sample=s['name'],capacity=c,arm=arm,method=LABELS[arm])
                measured=[r for r in rows if WARM<=r['simulated_start_ms']<END]
                mid,end=result['pressure_windows']
                summary=dict(**meta,**integrate(rows,arrivals,WARM,END,c),
                    preemptions=sum(r['preemptions'] for r in measured),
                    recomputed_tokens=sum(r['recomputed_tokens'] for r in measured),
                    completed_qps=(end['completed']-mid['completed'])/60,
                    backlog_growth_qps=(end['outstanding']-mid['outstanding'])/60,
                    waiting_first_end=end['waiting_first_token'],
                    arrivals_second_half=end['arrivals']-mid['arrivals'])
                summaries.append(summary)
                # Equal one-second bins; step integrals, never round-count means.
                for start in np.arange(WARM,END,1000):
                    series.append(dict(**meta,time_s=(start+500)/1000,
                                       **integrate(rows,arrivals,start,start+1000,c)))
    assert all(len(v)==1 for v in hashes.values()),'Inputs differ between methods/capacities'
    assert len(summaries)==128
    write_csv(OUT/'per_window.csv',summaries)
    write_csv(OUT/'timeseries_1s.csv',series)
    fields=[k for k in summaries[0] if k not in ('source','sample','capacity','arm','method')]
    grouped=[]
    for source in ('code','conversation'):
        for c in CAPS:
            for arm,*_ in METHODS:
                pair=[r for r in summaries if (r['source'],r['capacity'],r['arm'])==(source,c,arm)]
                assert len(pair)==2
                row=dict(source=source,capacity=c,arm=arm,method=LABELS[arm],n=2)
                for field in fields:
                    values=[r[field] for r in pair]
                    row.update({field:float(np.mean(values)),field+'_min':min(values),field+'_max':max(values)})
                grouped.append(row)
    write_csv(OUT/'summary.csv',grouped)
    return summaries,grouped,series

def export(fig,axes,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(FIG/f'{name}.alignment.json'),strict=True)
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.svg')
    fig.savefig(FIG/f'{name}.png',dpi=300)
    plt.close(fig)

def setup():
    FIG.mkdir(parents=True,exist_ok=True)
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})

def plot_capacity(rows):
    fig,axes=plt.subplots(2,4,figsize=(12,6.3))
    fig.subplots_adjust(left=.065,right=.985,bottom=.22,top=.92,wspace=.36,hspace=.5)
    metrics=(('mean_kv_pct','Mean KV occupancy (%)'),('sd_kv_pp','Temporal KV SD (pp)'),
             ('preemptions','Preemptions (60–120 s)'),('waiting_first_mean','Requests awaiting first token'))
    for i,source in enumerate(('code','conversation')):
        for j,(field,ylabel) in enumerate(metrics):
            ax=axes[i,j]
            for arm,label,color,ls,marker in METHODS:
                p=sorted([r for r in rows if r['source']==source and r['arm']==arm],key=lambda r:r['capacity'])
                y=np.array([r[field] for r in p]); lo=np.array([r[field+'_min'] for r in p]); hi=np.array([r[field+'_max'] for r in p])
                ax.errorbar(range(4),y,yerr=np.array([y-lo,hi-y]),label=label,color=color,
                    linestyle=ls,marker=marker,markersize=3.5,linewidth=1.8 if arm=='arrival' else 1.,
                    elinewidth=.6,capsize=1.5,alpha=1 if arm=='arrival' else .8,zorder=5 if arm=='arrival' else 2)
            ax.set_xticks(range(4),[str(c) for c in CAPS]);ax.tick_params(axis='x',labelsize=7)
            ax.set_ylim(bottom=0);ax.set_ylabel(ylabel);ax.set_title(source.capitalize())
            ax.set_xlabel('KV capacity (blocks)');ax.grid(axis='y',color='#e1e4e6',linewidth=.5)
            ax.annotate(chr(97+i*4+j),xy=(0,1),xycoords='axes fraction',xytext=(-27,13),
                textcoords='offset points',fontweight='bold',fontsize=10)
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',bbox_to_anchor=(.52,.045),ncol=4)
    export(fig,axes,'kv_capacity_utilization')

def plot_trajectories(rows):
    fig,axes=plt.subplots(2,2,figsize=(9.6,6.2),sharex=True,sharey=True)
    fig.subplots_adjust(left=.08,right=.98,bottom=.20,top=.92,wspace=.18,hspace=.35)
    for i,source in enumerate(('code','conversation')):
        for j,c in enumerate((768,4096)):
            ax=axes[i,j]
            for arm,label,color,ls,marker in METHODS:
                p=[r for r in rows if r['sample']==source+'_d1_w00' and r['capacity']==c and r['arm']==arm]
                assert len(p)==60
                ax.plot([r['time_s'] for r in p],[r['mean_kv_pct'] for r in p],label=label,
                    color=color,linestyle=ls,linewidth=1.8 if arm=='arrival' else 1.,alpha=1 if arm=='arrival' else .8,
                    zorder=5 if arm=='arrival' else 2)
            ax.set_title(f'{source.capitalize()} | {c} KV blocks');ax.set_ylim(0,103);ax.set_xlim(60,120)
            ax.grid(axis='y',color='#e1e4e6',linewidth=.5)
            ax.annotate(chr(97+i*2+j),xy=(0,1),xycoords='axes fraction',xytext=(-27,13),
                textcoords='offset points',fontweight='bold',fontsize=10)
    for ax in axes[1]:ax.set_xlabel('Time (s)')
    for ax in axes[:,0]:ax.set_ylabel('KV occupancy (%, 1 s mean)')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.52,.035),ncol=4)
    export(fig,axes,'kv_utilization_timeline')

def contrasts(raw):
    paired=[]
    for r in raw:
        if r['arm']=='arrival': continue
        ours=next(x for x in raw if x['arm']=='arrival' and x['sample']==r['sample'] and x['capacity']==r['capacity'])
        paired.append(dict(source=r['source'],sample=r['sample'],capacity=r['capacity'],baseline=r['arm'],
            **{k+'_spare_minus_baseline':ours[k]-r[k] for k in ('mean_kv_pct','sd_kv_pp','preemptions','waiting_first_mean','completed_qps')}))
    write_csv(OUT/'paired_differences.csv',paired)

if __name__=='__main__':
    raw,summary,series=assemble()
    contrasts(raw);setup();plot_capacity(summary);plot_trajectories(series)
    for r in summary:
        if r['capacity']==768:
            print(r['source'],r['method'],*[f'{r[k]:.3f}' for k in ('mean_kv_pct','sd_kv_pp','preemptions','waiting_first_mean','completed_qps')])
