"""Independent decision reconstruction and complete-episode accounting."""
import math,statistics as st
from research.v1_budget_20260922.policy import State,project

def audit(result,rounds):
    assert result['status']=='complete' and result['completed']==len(result['requests'])
    cfg=result['config'];T,K=cfg['budget'],cfg['chunk'];mode=cfg['threshold']['candidate_mode']
    small=min(64,K) if mode=='fixed64' else max(1,K//(16 if mode=='ratio16' else 4))
    assert mode in ('fixed64','ratio16','ratio4')
    candidates=[(max(1,T//4),K),(max(1,T//2),K),(T,small),(max(1,T//2),small)]
    events=[];high={};recomputed=0;different=0;counts={str(i):0 for i in range(-1,4)}
    from research.v1_broad12h_20260925.policy import select as legacy
    for row in rounds:
        states=[State(**r) for r in row['state']]
        assert row['free_before']+sum(r.blocks for r in states)==row['usable_blocks']==row['total_blocks']-1
        events+=row['arrival_events'];now=row['features']['arrival_time_s']
        assert all(e['time_s']<=now for e in events)
        rate=sum(e['prompt']*math.exp(-(now-e['time_s'])) for e in events)
        assert math.isclose(rate,row['features']['arrival_rate'],rel_tol=1e-9,abs_tol=1e-7)
        work=sum(max(0,r.prompt-r.computed) for r in states if r.outputs==0)
        n=max(1,min(100000,math.ceil(8+128*(row['free_before']/(1+(work+rate)/16)))))
        assert n==row['threshold']
        args=(row['free_before'],16,32,8176);base=project(states,T,K,*args)
        expected=base;b,k=T,K;chosen=-1
        if base is not None:
            protected={r.rid:base.tokens[r.rid] for r in states if r.rid in base.tokens and
                (r.decode or (r.outputs==0 and base.tokens[r.rid]==r.remaining))}
            if base.growth>0 and len(states)>=2:
                for i,(tb,kb) in enumerate(candidates):
                    p=project(states,tb,kb,*args)
                    if p is not None and p.tokens and base.growth-p.growth>=n and all(p.tokens.get(rid,0)>=v for rid,v in protected.items()):
                        expected=p;b,k=tb,kb;chosen=i;break
            assert all(row['scheduled'].get(rid,0)>=v for rid,v in protected.items())
        assert (b,k)==(row['budget'],row['chunk'])
        assert chosen==row['features']['selected_candidate'];counts[str(chosen)]+=1
        if expected is not None:
            assert expected.tokens==row['scheduled'] and expected.growth==row['actual_growth_blocks'] and row['preemptions']==0
        if row['intervened']:assert base.growth-row['actual_growth_blocks']>=n
        assert row['free_before']-row['free_after']==row['actual_growth_blocks']
        assert sum(row['scheduled'].values())<=b
        ref=legacy(states,T,K,row['free_before'],row['total_blocks'],16,32,8176,
                   cfg['threshold'],dict(arrival_rate=rate))
        if ref['selected'] is not None:
            different+=ref['selected'].tokens!=row['scheduled']
            if mode=='fixed64' or small==min(64,K):assert ref['selected'].tokens==row['scheduled']
        for a in row['allocations']:
            recomputed+=min(a['tokens'],max(0,high.get(a['id'],0)-a['before']))
            high[a['id']]=max(high.get(a['id'],0),a['before']+a['tokens'])
    unique=sum(r['prompt']+r['output']-1 for r in result['requests'])
    executed=sum(sum(r['scheduled'].values()) for r in rounds)
    assert executed-unique==recomputed==sum(r['recomputed_tokens'] for r in rounds)
    ttft=[]
    for r in result['requests']:
        c=result['clocks'][r['id']]
        assert c['tokens']==r['output']==len(c['token_times_ms'])
        assert c['first_ms']==c['token_times_ms'][0]>=r['arrival_ms']
        assert c['finish_ms']>=c['token_times_ms'][-1]
        ttft.append(c['first_ms']-r['arrival_ms'])
    assert math.isclose(st.fmean(ttft),result['metrics']['mean_ttft_ms'],rel_tol=1e-10)
    assert len(events)==len(result['requests'])
    return dict(passed=True,rounds=len(rounds),recomputed_tokens=recomputed,
                different_from_fixed64=different,selected_candidates=counts)
