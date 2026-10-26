"""Figures for the Chinese evaluation draft; archived results only."""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import json
import statistics as st
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
GPU = ROOT/'writing/www/experiments/gpu_baselines_4090_20260930_r3'
SIM = ROOT/'artifacts/www_evaluation_completion_20261003_r1'
OUT = ROOT/'writing/www/figures/evaluation_draft_20261003'
LABEL = dict(arrival='SPARE', basic='w/o arrival', srp='SRP', native='Native V1',
    sarathi='Sarathi rule', past_future='Past-Future rule',
    ssjf_oracle='SSJF (O)', mc_sf_oracle='MC-SF (O)', trail_oracle='TRAIL order (O)',
    tie_oracle_proxy='TIE (P)', uniboost_static='UniBoost priority',
    apt_serve_kv_rule='Apt-Serve KV density', sola_rule='SOLA risk order')
ORDER=list(LABEL)
plt.rcParams.update({'font.size':9, 'axes.titlesize':10, 'axes.spines.top':False,
    'axes.spines.right':False, 'pdf.fonttype':42, 'ps.fonttype':42})


def read_csv(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def baseline_fig(data, name, specs):
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,(key,scale,title,unit) in zip(axes.flat,specs):
        vals=[float(data[a][key])*scale for a in ORDER]
        bars=ax.barh(range(len(ORDER)),vals,color=['#2878a8' if a=='arrival' else '#86b3ca' if a=='basic' else '#da9254' if a=='past_future' else '#bdc3c7' for a in ORDER])
        for a,b in zip(ORDER,bars):
            if 'oracle' in a:b.set_hatch('///')
        ax.set_yticks(range(len(ORDER)),[LABEL[a] for a in ORDER],fontsize=8)
        ax.invert_yaxis();ax.set_title(title);ax.set_xlabel(unit)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('GPU measurements: 16 scenarios, four repeats per method\nScenario-weighted means; O/P use realized output lengths',fontsize=11)
    save(fig,name)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    agg={r['arm']:r for r in read_csv(GPU/'analysis/aggregate.csv')}
    baseline_fig(agg,'E1_response',[
        ('ttft_ms_mean',.001,'Mean TTFT','Seconds (lower is better)'),
        ('ttft_ms_p99',.001,'P99 TTFT','Seconds (lower is better)'),
        ('e2e_ms_mean',.001,'Mean E2E','Seconds (lower is better)'),
        ('e2e_ms_p99',.001,'P99 E2E','Seconds (lower is better)')])
    baseline_fig(agg,'E2_service',[
        ('tpot_ms_mean',1,'Mean per-request TPOT','Milliseconds (lower is better)'),
        ('max_itl_ms_p99',1,'P99 of per-request maximum ITL','Milliseconds (lower is better)'),
        ('output_throughput_tps',1,'Output throughput (finite replay)','Tokens/s (higher is better)'),
        ('slo_ttft5000_maxitl200_attainment',100,'TTFT <= 5 s AND max ITL <= 200 ms','Attainment % (higher is better)')])
    baseline_fig(agg,'E3_resources',[
        ('peak_kv_util',100,'Peak KV utilization','% of usable capacity'),
        ('preemptions',1,'Preemption events','Mean count per run'),
        ('recomputed_tokens',.001,'Executed recomputation','Mean thousands of tokens per run'),
        ('schedule_ms_mean',1,'Scheduling time (includes auditing)','Milliseconds per iteration')])
    pairs=read_csv(SIM/'paired_comparisons.csv')
    metrics=read_csv(SIM/'metrics.csv')
    profiles=['resource','resource_staircase','gpu_fitted']
    titles=['Resource simulation','Staircase simulation','Fitted simulation']
    colors={'srp':'#777777','fixed256':'#aa6d3b','constant16':'#a680b8','basic':'#86b3ca','no_finish_guard':'#5b9b7b'}
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for j,(profile,title) in enumerate(zip(profiles,titles)):
        for control in ['srp','fixed256','constant16']:
            yy=[st.fmean(float(r['spare_reduction_pct']) for r in pairs if r['family']=='core' and r['profile']==profile and r['control']==control and int(r['capacity'])==c) for c in [768,2048,4096]]
            axes[0,j].plot([768,2048,4096],yy,'o-',label=control,color=colors[control])
        for arm in ['spare','basic','srp','fixed512']:
            yy=[st.fmean(float(r['ttft_ms_mean'])/1000 for r in metrics if r['family']=='sustained' and r['profile']==profile and r['arm']==arm and int(r['capacity'])==2048 and float(r['arrival_rps'])==rate) for rate in [2,5,10,20]]
            axes[1,j].plot([2,5,10,20],yy,'o-',label={'spare':'SPARE','basic':'w/o arrival','srp':'SRP','fixed512':'fixed512'}[arm],color={'spare':'#2878a8','basic':'#86b3ca','srp':'#777777','fixed512':'#aa6d3b'}[arm])
        axes[0,j].axhline(0,color='black',lw=.7);axes[0,j].set_title(title)
        axes[0,j].set_xlabel('Configured KV blocks');axes[0,j].set_ylabel('SPARE TTFT reduction (%)')
        axes[1,j].set_xlabel('Offered requests/s');axes[1,j].set_ylabel('Mean TTFT (s), 2048 blocks')
        for ax in axes[:,j]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Six source windows; profiles remain separate. Negative reductions mean SPARE is slower.')
    save(fig,'E4_capacity_load')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    controls=['srp','basic','no_finish_guard','constant16','constant32','fixed256','fixed512','occupancy70']
    for ax,profile,title in zip(axes,profiles,titles):
        for idx,c in enumerate(controls):
            rr=[r for r in pairs if r['family']=='core' and r['profile']==profile and r['control']==c]
            values=[st.fmean(float(r['spare_reduction_pct']) for r in rr if r['window']==w) for w in sorted({r['window'] for r in rr})]
            ax.scatter(values,[idx]*len(values),s=16,alpha=.65,color='#2878a8')
            ax.plot(st.fmean(values),idx,'|',color='black',ms=13,mew=2)
        ax.set_yticks(range(len(controls)),controls);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.8)
        ax.set_title(title);ax.set_xlabel('SPARE TTFT reduction (%)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Ablations and controls: six window means; black mark is their mean')
    save(fig,'E5_ablations')
    # Selection is explicit and uses the complete tight-capacity comparison:
    # closest to the median positive/negative effect; repeat zero for both arms.
    runs=read_csv(GPU/'analysis/runs.csv');by=defaultdict(list)
    for r in runs:by[(r['cell'],r['arm'])].append(r)
    effects={c:100*(1-st.fmean(float(r['ttft_ms_mean']) for r in by[c,'arrival'])/st.fmean(float(r['ttft_ms_mean']) for r in by[c,'srp'])) for c,a in by if a=='arrival' and not c.endswith('c4096')}
    cell_names=sorted({c for c,a in by})
    controls=[a for a in ORDER if a!='arrival']
    matrix=[[100*(1-st.fmean(float(r['ttft_ms_mean']) for r in by[c,'arrival'])/st.fmean(float(r['ttft_ms_mean']) for r in by[c,a])) for a in controls] for c in cell_names]
    fig,ax=plt.subplots(figsize=(12,7),layout='constrained')
    bound=max(abs(v) for line in matrix for v in line)
    im=ax.imshow(matrix,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
    ax.set_xticks(range(len(controls)),[LABEL[a] for a in controls],rotation=45,ha='right')
    ax.set_yticks(range(len(cell_names)),[c.split('_',1)[1].replace('_',' ') for c in cell_names],fontsize=8)
    for i,line in enumerate(matrix):
        for j,v in enumerate(line):ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=7,color='white' if abs(v)>bound*.6 else 'black')
    ax.set_title('GPU: per-scenario SPARE mean TTFT reduction (%)\nFour repeats averaged before pairing; negative values favor the comparison method')
    fig.colorbar(im,ax=ax,label='TTFT reduction (%)')
    save(fig,'E1b_scenarios')
    chosen=[]
    for positive in [True,False]:
        keys=[c for c,v in effects.items() if (v>0)==positive]
        median=st.median(effects[c] for c in keys)
        chosen.append(min(keys,key=lambda c:(abs(effects[c]-median),c)))
    selection=[];fig,axes=plt.subplots(3,2,figsize=(11,8),layout='constrained',sharex='col')
    for j,cell in enumerate(chosen):
        selection.append(dict(cell=cell,all_repeat_effect_pct=effects[cell],repeat=0))
        for arm,color in [('arrival','#2878a8'),('srp','#777777')]:
            row=next(r for r in by[cell,arm] if int(r['repeat'])==0)
            path=GPU/row['result_path'];d=json.loads(path.read_text())
            events=defaultdict(int)
            for r in d['requests']:events[r['arrival_ms']]+=1;events[r['first_ms']]-=1
            tt=sorted(events);count=0;values=[]
            for t in tt:count+=events[t];values.append(count)
            area=sum((tt[i+1]-tt[i])*values[i] for i in range(len(tt)-1))
            assert math.isclose(area,sum(r['ttft_ms'] for r in d['requests']),rel_tol=1e-10)
            axes[0,j].step([t/1000 for t in tt],values,where='post',color=color,label=LABEL[arm])
            with gzip.open(path.parent.parent/d['rounds_path'],'rt') as f:rr=json.load(f)
            starts=[(r['step_end_ms']-r['step_ms'])/1000 for r in rr]
            axes[1,j].step(starts,[r['free_before'] for r in rr],where='post',color=color)
            total=0;recompute=[]
            for r in rr:total+=r['recomputed_tokens'];recompute.append(total/1000)
            axes[2,j].step([r['step_end_ms']/1000 for r in rr],recompute,where='post',color=color)
        axes[0,j].set_title(cell.replace('cell','').replace('_',' ')+'\nExplanatory case, repeat 0',fontsize=9)
        axes[0,j].legend();axes[0,j].set_ylabel('Requests awaiting first token')
        axes[1,j].set_ylabel('Free KV blocks before scheduling')
        axes[2,j].set_ylabel('Cumulative recomputation (k tokens)')
        for ax in axes[:,j]:ax.set_xlabel('Time since replay start (s)');ax.grid(alpha=.2)
    save(fig,'E6_waiting_mechanism')
    (OUT/'case_selection.json').write_text(json.dumps(dict(rule='Among tight GPU scenarios, choose closest to median positive/negative four-repeat mean effect vs SRP; tie by cell ID. Plot repeat 0; descriptive post-hoc cases.',cases=selection),indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    params=['n0_4','n0_16','alpha64','alpha256','H05','H2','tau05','tau2']
    for ax,profile,title in zip(axes,profiles,titles):
        for idx,c in enumerate(params):
            rr=[r for r in pairs if r['family']=='sensitivity' and r['profile']==profile and r['control']==c]
            values=[st.fmean(float(r['spare_reduction_pct']) for r in rr if r['window']==w) for w in sorted({r['window'] for r in rr})]
            ax.scatter(values,[idx]*len(values),s=16,alpha=.65,color='#2878a8');ax.plot(st.fmean(values),idx,'|',color='black',ms=13,mew=2)
        ax.set_yticks(range(len(params)),params);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.8)
        ax.set_title(title);ax.set_xlabel('Default SPARE TTFT reduction (%)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Parameter sensitivity: six window means; negative values favor the parameter variant')
    save(fig,'E7_sensitivity')
    print(json.dumps(dict(figures=8,cases=selection,output=str(OUT))))


if __name__=='__main__':main()
