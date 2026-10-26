"""Post-run metadata validation, stratification and same-state action accounting."""
import collections,gzip,math,statistics as st,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_ratio_closed_loop_20260928.common import *
from research.v1_budget_20260922.policy import State
from research.v1_broad12h_20260925.policy import select as legacy

def summarize(rs):
    return dict(scenes=len(rs),mean_improvement_pct=st.fmean(r['improvement_pct'] for r in rs),
        min_improvement_pct=min(r['improvement_pct'] for r in rs),max_improvement_pct=max(r['improvement_pct'] for r in rs),
        positive=sum(r['improvement_pct']>0 for r in rs),both_positive=sum(min(r['repeat_improvement_pct'])>0 for r in rs),
        both_negative=sum(max(r['repeat_improvement_pct'])<0 for r in rs),
        changed_rounds=sum(r['changed_rounds'] for r in rs),rounds=sum(r['rounds'] for r in rs),
        with_changed_action=sum(r['changed_rounds']>0 for r in rs),
        preemptions_change_pct=100*(sum(r['arm_preemptions'] for r in rs)/max(1,sum(r['baseline_preemptions'] for r in rs))-1),
        recomputed_change_pct=100*(sum(r['arm_recomputed'] for r in rs)/max(1,sum(r['baseline_recomputed'] for r in rs))-1))

d=verify();summary=read(OUT/'analysis/summary.json');rs=summary['comparisons'];groups=[];actions=[];aa=[]
for arm in ARMS[1:]:
    selected=[r for r in rs if r['arm']==arm]
    for label,fields in [('config',('budget','chunk')),('source',('source',)),('shape',('shape',)),('source_config',('source','budget','chunk'))]:
        for key in sorted({tuple(r[f] for f in fields) for r in selected}):
            sub=[r for r in selected if tuple(r[f] for f in fields)==key]
            groups.append(dict(arm=arm,stratum=label,**dict(zip(fields,key)),**summarize(sub)))
    for scope,sub in [('all',selected),('changed_configs',[r for r in selected if not r['same_candidates']]),
                      ('equivalent_controls',[r for r in selected if r['same_candidates']])]:
        if sub:groups.append(dict(arm=arm,stratum=scope,**summarize(sub)))
for g in d['groups']:
    if g['key'] in summary['excluded']:continue
    folder=OUT/'groups'/g['key'];aa.append(dict(group=g['key'],**read(folder/'aa.json')))
    for p in sorted((folder/'runs').glob('*.json')):
        r=read(p);cfg=r['config'];assert (cfg['budget'],cfg['chunk'],cfg['concurrency'])==(g['budget'],g['chunk'],32)
        assert cfg['threshold']==spec(r['arm'])
        assert r['repeat'] in (0,1) and r['shape'] in SHAPES
        with gzip.open(OUT/r['rounds_path'],'rt') as f:rows=json.load(f)
        assert all(row['total_blocks']==g['capacity'] for row in rows)
        assert all(row['round']==i for i,row in enumerate(rows))
        assert r['metrics']['round_count']==len(rows)
        assert r['metrics']['preemptions']==sum(row['preemptions'] for row in rows)
        for metric,key in [('schedule_ms_mean','schedule_ms'),('step_ms_mean','step_ms')]:
            assert math.isclose(r['metrics'][metric],st.fmean(row[key] for row in rows),rel_tol=1e-10)
        if r['phase']!='formal':continue
        counter=collections.Counter();deltas=collections.Counter()
        for row in rows:
            counter['all_rounds']+=1
            counter['selected_'+str(row['features']['selected_candidate'])]+=1
            counter['q_'+str(sum(row['scheduled'].values()))]+=1
            counter['fallback_rounds']+=row['recovery_fallback']
            counter['executed_tokens']+=sum(row['scheduled'].values())
            counter['schedule_us']+=round(row['schedule_ms']*1000)
            counter['step_us']+=round(row['step_ms']*1000)
            states=[State(**s) for s in row['state']]
            ref=legacy(states,g['budget'],g['chunk'],row['free_before'],g['capacity'],16,32,8176,
                       cfg['threshold'],dict(arrival_rate=row['features']['arrival_rate']))['selected']
            if ref is None or ref.tokens==row['scheduled']:continue
            counter['changed_rounds']+=1
            dq=sum(row['scheduled'].values())-sum(ref.tokens.values());dg=row['actual_growth_blocks']-ref.growth
            deltas['q_delta_sum']+=dq;deltas['growth_delta_sum']+=dg
            for name,v in [('q',dq),('growth',dg)]:counter[name+('_less' if v<0 else '_more' if v>0 else '_same')]+=1
            for s in states:
                delta=row['scheduled'].get(s.rid,0)-ref.tokens.get(s.rid,0)
                deltas['pending_tokens_delta' if s.outputs==0 else 'output_started_tokens_delta']+=delta
        assert counter['changed_rounds']==r['audit']['different_from_fixed64']
        actions.append(dict(group=g['key'],source=g['source'],window=g['window'],budget=g['budget'],chunk=g['chunk'],
            shape=r['shape'],arm=r['arm'],repeat=r['repeat'],counts=dict(counter),deltas=dict(deltas)))
write(OUT/'analysis/stratified.json',groups)
write(OUT/'analysis/action_details.json',actions)
write(OUT/'analysis/extra_validation.json',dict(passed=True,formal_runs=len(actions),aa=aa,
    checks=['group configuration','capacity','sequential round indices','metrics from rows','same-state action counts']))
for r in groups:
    if r['stratum'] in ('config','all','changed_configs','equivalent_controls'):print(r)
