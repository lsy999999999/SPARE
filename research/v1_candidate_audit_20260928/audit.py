"""Independent reconstruction of physical plans, clocks and waiting labels."""
import math
import statistics as st
from research.v1_candidate_audit_20260928.common import *
from research.v1_dynamic_actions_20260927.policy import select
from research.v1_budget_20260922.policy import State


def audit(result, rounds, anchor=None):
    assert result['status']=='complete' and result['completed']==len(result['requests'])
    config=result['config']; branch=config.get('branch'); events=[]
    high={}; recomputed=0
    for row in rounds:
        states=[State(**r) for r in row['state']]
        assert row['free_before']+sum(r.blocks for r in states)==row['usable_blocks']
        events+=row['arrival_events']; now=row['features']['arrival_time_s']
        assert all(e['time_s']<=now for e in events)
        rate=sum(e['prompt']*math.exp(-(now-e['time_s'])) for e in events)
        assert math.isclose(rate,row['features']['arrival_rate'],rel_tol=1e-9,abs_tol=1e-7)
        threshold=config['threshold']
        if branch and row['round']==branch['round']:threshold=branch['threshold']
        elif branch and row['round']>branch['round']:threshold=branch['after_threshold']
        choice=select(states,config['budget'],config['chunk'],row['free_before'],row['total_blocks'],
                      16,32,8176,threshold,{'arrival_rate':rate})
        assert choice['threshold']==row['threshold']
        assert (choice['budget'],choice['chunk'])==(row['budget'],row['chunk'])
        if choice['selected'] is not None:
            assert choice['selected'].tokens==row['scheduled']
            assert choice['selected'].growth==row['actual_growth_blocks']
            assert row['preemptions']==0
        assert row['free_before']-row['free_after']==row['actual_growth_blocks']
        assert sum(row['scheduled'].values())<=row['budget']
        if row['intervened']:
            assert row['baseline_growth_blocks']-row['actual_growth_blocks']>=row['threshold']
        for a in row['allocations']:
            recomputed+=min(a['tokens'],max(0,high.get(a['id'],0)-a['before']))
            high[a['id']]=max(high.get(a['id'],0),a['before']+a['tokens'])
    unique=sum(r['prompt']+r['output']-1 for r in result['requests'])
    executed=sum(sum(r['scheduled'].values()) for r in rounds)
    assert executed-unique==sum(r['recomputed_tokens'] for r in rounds)==recomputed
    ttft=[]
    for r in result['requests']:
        c=result['clocks'][r['id']]
        assert c['tokens']==r['output']==len(c['token_times_ms'])
        assert c['first_ms']==c['token_times_ms'][0]>=r['arrival_ms']
        assert c['finish_ms']>=c['token_times_ms'][-1]
        ttft.append(c['first_ms']-r['arrival_ms'])
    assert math.isclose(st.fmean(ttft),result['metrics']['mean_ttft_ms'],rel_tol=1e-10)
    if branch:
        index=branch['round']; assert anchor is not None
        assert sum(r['branch_applied'] for r in rounds)==1
        assert all(r['state_fingerprint']==anchor[i]['state_fingerprint'] for i,r in enumerate(rounds[:index+1]))
        assert all(r['scheduled']==anchor[i]['scheduled'] for i,r in enumerate(rounds[:index]))
        assert rounds[index]['scheduled']==result['action']['tokens']
        label=measure(result,rounds,index)
        if 'label' in result:
            for k in ('area_per_request_ms','first_round_ms','later_ms'):
                assert math.isclose(label[k],result['label'][k],rel_tol=1e-10,abs_tol=1e-7)
    return dict(passed=True,rounds=len(rounds),recomputed_tokens=recomputed)


def measure(result,rounds,index):
    row=rounds[index]; start=(row['probe_decision_wall']-result['start_wall'])*1000
    pending=[r['rid'] for r in row['state'] if r['outputs']==0]
    assert pending
    count=len(result['requests']); end=row['step_end_ms']
    area=sum(result['clocks'][rid]['first_ms']-start for rid in pending)/count
    first=len(pending)*(end-start)/count
    later=sum(max(0,result['clocks'][rid]['first_ms']-end) for rid in pending)/count
    assert math.isclose(area,first+later,rel_tol=1e-10,abs_tol=1e-7)
    tail=rounds[index:]
    release=None;preemption=None
    for i,r in enumerate(tail):
        if r['preemptions'] and preemption is None:
            preemption=dict(round_offset=i,time_ms=r['step_end_ms']-start,count=r['preemptions'])
        if i and r['free_before']>tail[i-1]['free_after'] and release is None:
            release=dict(round_offset=i,time_ms=tail[i-1]['step_end_ms']-start,
                         blocks=r['free_before']-tail[i-1]['free_after'])
    return dict(area_per_request_ms=area,first_round_ms=first,later_ms=later,pending=len(pending),
        step_ms=row['step_ms'],schedule_ms=row['schedule_ms'],q=sum(row['scheduled'].values()),
        growth=row['actual_growth_blocks'],free_after=row['free_after'],
        preemptions=sum(r['preemptions'] for r in tail),recomputed_tokens=sum(r['recomputed_tokens'] for r in tail),
        kv_block_ms_approx=sum((r['usable_blocks']-r['free_after'])*r['step_ms'] for r in tail),
        min_free=min(r['free_after'] for r in tail),first_release=release,first_preemption=preemption,
        first_order=sorted(pending,key=lambda rid:(result['clocks'][rid]['first_ms'],rid)))
