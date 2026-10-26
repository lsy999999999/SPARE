"""Independent arithmetic audit and compact comparison figures (no fitting)."""
import csv
import gzip
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_simple_formula_20260924.common import *


def audit(r):
    assert r['status']=='complete'
    assert digest(OUT/r['rounds_path'])==r['rounds_sha256']
    ttft=[];e2e=[]
    for req in r['requests']:
        c=r['clocks'][req['id']]
        assert c['tokens']==req['output']==len(c['token_times_ms'])
        assert c['first_ms']==c['token_times_ms'][0]
        assert c['finish_ms']>=c['token_times_ms'][-1]
        assert c['arrival_ms']==req['arrival_ms']<=c['first_ms']
        ttft.append(c['first_ms']-c['arrival_ms']);e2e.append(c['finish_ms']-c['arrival_ms'])
    assert math.isclose(st.fmean(ttft),r['metrics']['mean_ttft_ms'],abs_tol=1e-6)
    assert math.isclose(st.fmean(e2e),r['metrics']['mean_e2e_ms'],abs_tol=1e-6)
    with gzip.open(OUT/r['rounds_path'],'rt') as f:rounds=json.load(f)
    verified=0;decisions=0;different=0;eligible=0;blocked=0
    for row in rounds:
        assert 0<=row['free_after']<=row['total_blocks']-1
        if 'state' not in row:continue
        state={s['rid']:s for s in row['state']}
        rem=[max(0,s['prompt']-s['computed']) for s in state.values() if s['outputs']==0]
        x=row['free_before']/(1+sum(rem)/16)
        assert math.isclose(x,row['features']['work_free_ratio'],abs_tol=1e-12)
        mean=st.fmean(rem) if rem else 0
        cv=math.sqrt(st.fmean((v-mean)**2 for v in rem))/mean if mean else 0
        assert math.isclose(cv,row['features']['remaining_cv'],abs_tol=1e-12)
        spec=r['job']['config']['threshold']
        if isinstance(spec,dict):
            n=max(1,min(100000,math.ceil(spec['intercept']+spec['coefficients'][0]*x+spec['coefficients'][1]*cv)))
            assert n==row['threshold']
        else:assert row['threshold']==(None if spec=='off' else spec)
        if row['baseline'] is None:continue
        verified+=1
        def growth(tokens):
            return sum(max(0,(state[rid]['computed']+count+15)//16-state[rid]['blocks']) for rid,count in tokens.items())
        assert growth(row['baseline'])==row['baseline_growth_blocks']
        assert growth(row['scheduled'])==row['free_before']-row['free_after']
        protected={rid:n for rid,n in row['baseline'].items() if
                   (state[rid]['outputs']>0 and state[rid]['tokens']-state[rid]['computed']==1)
                   or (state[rid]['outputs']==0 and n==state[rid]['tokens']-state[rid]['computed'])}
        assert all(row['scheduled'].get(rid,0)>=n for rid,n in protected.items())
        choices=[]
        for c in row['candidate_diagnostics']:
            if c['tokens'] is None:continue
            assert c['growth']==growth(c['tokens'])
            assert c['saving']==growth(row['baseline'])-c['growth']
            assert c['guard']==all(c['tokens'].get(rid,0)>=n for rid,n in protected.items())
            if c['guard']:choices.append(c)
        def action(n):
            if n is not None and len(state)>=2 and row['baseline_growth_blocks']>0:
                for c in choices:
                    if c['saving']>=n:return c['tokens']
            return row['baseline']
        assert row['scheduled']==action(row['threshold'])
        if row['features']['pending_prefill']:
            decisions+=1;different+=action(16)!=row['scheduled']
            eligible+=action(1)!=row['baseline']
            blocked+=(action(1)!=row['baseline'] and row['scheduled']==row['baseline'])
    for name,field in [('preemptions','preemptions'),('recomputed_tokens','recomputed_tokens'),('interventions','intervened')]:
        assert r['metrics'][name]==sum(row[field] for row in rounds)
    return dict(rounds=len(rounds),verified=verified,prefill_decisions=decisions,
                differs_from_n16=different,positive_saving_opportunities=eligible,blocked_opportunities=blocked)


def main():
    d=read(OUT/'design.json');frozen=read(OUT/'frozen_formula.json');m=read(OUT/'manifest.json')
    assert read(OUT/'status.json')['state']=='complete'
    assert not list((OUT/'contamination').glob('*.json'))
    for name,h in d['trace_hashes'].items():assert object_digest(read(OUT/f'inputs/{name}.json'))==h
    for name,h in m['sources'].items():assert digest(OUT/'source'/name)==h
    runs=[read(p) for p in sorted((OUT/'runs').glob('*.json'))]
    result=[];audits=[];groups={}
    for r in runs:
        j=r['job'];a=audit(r);audits.append(dict(key=j['key'],**a))
        if j['phase']=='aa':continue
        assert object_digest(r['requests'])==d['trace_hashes'][j['case']]
        assert r['gpu']==m['bindings'][str(j['capacity'])]
        if j['phase']=='holdout':
            assert r['saved_at']>frozen['time']
            if j['arm']=='formula':assert j['config']['threshold']==frozen['spec']
        row=dict(phase=j['phase'],case=j['case'],capacity=j['capacity'],arm=j['arm'],repeat=j['repeat'],gpu=r['gpu'],
            **{k:v for k,v in r['metrics'].items() if not isinstance(v,list)},**a)
        result.append(row)
        groups.setdefault((j['phase'],j['case'],j['capacity']),{}).setdefault(j['arm'],[]).append(row)
    pairs=[]
    for (phase,case,cap),arms in sorted(groups.items()):
        if phase=='holdout':assert set(arms)==set(d['holdout_methods'])
        def avg(arm,key='mean_ttft_ms'):return st.fmean(x[key] for x in arms[arm])
        baseline=avg('off')
        for arm in arms:
            rec=dict(phase=phase,case=case,capacity=cap,arm=arm,ttft_ms=avg(arm),
                     gain_off_pct=100*(1-avg(arm)/baseline),gain_n16_pct=100*(1-avg(arm)/avg('n16')),
                     repeats=len(arms[arm]),e2e_ms=avg(arm,'mean_e2e_ms'),tbt_ms=avg(arm,'mean_tbt_ms'),
                     p95_ttft_ms=avg(arm,'p95_ttft_ms'),preemptions=avg(arm,'preemptions'),
                     recomputed_tokens=avg(arm,'recomputed_tokens'),interventions=avg(arm,'interventions'))
            if phase=='holdout':
                best=min(avg(a) for a in ('off','n16','n32'))
                rec.update(gain_native_pct=100*(1-avg(arm)/avg('native')),gap_best_fixed_pct=100*(avg(arm)/best-1),
                           e2e_gain_off_pct=100*(1-avg(arm,'mean_e2e_ms')/avg('off','mean_e2e_ms')),
                           tbt_gain_off_pct=100*(1-avg(arm,'mean_tbt_ms')/avg('off','mean_tbt_ms')))
            pairs.append(rec)
    assert len([r for r in runs if r['job']['phase']=='holdout'])==50
    for filename,rows in [('runs.csv',result),('comparisons.csv',pairs),('audit_rounds.csv',audits)]:
        p=OUT/'analysis'/filename;p.parent.mkdir(parents=True,exist_ok=True)
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    held=[r for r in pairs if r['phase']=='holdout' and r['arm']=='formula']
    summary=dict(frozen=frozen,runs=len(runs),gpu_formal_runs=len(result),
        audit=dict(passed=True,rounds=sum(a['rounds'] for a in audits),verified=sum(a['verified'] for a in audits),
                   request_completions=sum(len(r['requests']) for r in runs),holdout_cells=len(held)),
        holdout=held,development=[r for r in pairs if r['phase']=='dev'])
    write(OUT/'analysis/summary.json',summary)
    plot(pairs)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('development',)},ensure_ascii=False,indent=2))


def plot(pairs):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/simple-formula-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(formula='#007f73',n16='#dd8833',n32='#6875b8',native='#a3a3a3')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
    for ax,source in zip(axes,('code','conversation')):
        for arm in ('formula','n16','n32'):
            rs=sorted([r for r in pairs if r['phase']=='holdout' and r['case']==source+'_holdout' and r['arm']==arm],key=lambda r:r['capacity'])
            ax.plot([r['capacity'] for r in rs],[r['gain_off_pct'] for r in rs],marker='o',label=arm,color=colors[arm])
        ax.axhline(0,color='black',lw=.7);ax.set_title(source+' (new minute, 64 requests)')
        ax.set_xlabel('KV capacity (blocks)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean TTFT improvement over off (%)')
    fig.suptitle('Frozen formula: complete GPU windows; single-run holdout')
    fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/capacity_gain.png',dpi=180);fig.savefig(OUT/'figures/capacity_gain.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
    for ax,source in zip(axes,('code','conversation')):
        for arm in ('off','n16','formula','native'):
            rs=sorted([r for r in pairs if r['phase']=='holdout' and r['case']==source+'_holdout' and r['arm']==arm],key=lambda r:r['capacity'])
            ax.plot([r['capacity'] for r in rs],[r['ttft_ms']/1000 for r in rs],marker='o',label=arm,color=colors.get(arm,'black'))
        ax.set_title(source);ax.set_xlabel('KV capacity (blocks)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean TTFT (seconds)')
    fig.suptitle('FCFS native vs reordered policies (same configuration, instrumented)')
    fig.tight_layout();fig.savefig(OUT/'figures/native_comparison.png',dpi=180);plt.close(fig)
    held=[r for r in pairs if r['phase']=='holdout' and r['arm']=='formula']
    labels=[r['case'].replace('_holdout','')+' / '+str(r['capacity']) for r in held]
    fig,ax=plt.subplots(figsize=(10,4.5));ax.barh(labels,[r['gap_best_fixed_pct'] for r in held],color='#007f73')
    ax.axvline(0,color='black',lw=.7);ax.set_xlabel('TTFT gap to best measured fixed threshold {off, 16, 32} (%)')
    ax.set_title('Negative = dynamic formula better; not a theoretical optimum');fig.tight_layout()
    fig.savefig(OUT/'figures/fixed_gap.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
