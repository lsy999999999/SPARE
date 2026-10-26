"""Finite measured budgets, unchanged arrival threshold and physical safeguards."""
from bisect import bisect_left
from functools import lru_cache
from research.v1_efficient_points_20260927.common import OUT, read
from research.v1_dynamic_actions_20260927.policy import select as reference_select
from research.v1_completion_order_20260927.policy import completion_work, remaining_work
from research.v1_budget_20260922.policy import project


@lru_cache(maxsize=1)
def calibration():
    return read(OUT/'calibration.json')


def predicted_ms(rows, plan, chunk, table):
    # Actual scheduled q, not nominal budget. Right bracket preserves measured
    # 256->257 discontinuities. Below-grid plans use the first measured point.
    q=sum(plan.tokens.values())
    active=[r for r in rows if plan.tokens.get(r.rid,0)>0]
    context=sum(plan.tokens[r.rid]*r.computed for r in active)/max(1,q)
    decode=any(r.decode for r in active)
    profiles=['mixed'] if decode else ['fresh','long']
    profile=min(profiles,key=lambda p:abs(table['contexts'][p]-context))
    curve=table['curves'][f'{profile}_k{chunk}']
    pos=min(bisect_left([r['q'] for r in curve],q),len(curve)-1)
    return curve[pos]['ms'],profile,curve[pos]['q']


def select(rows,budget,chunk,free,total,block_size,concurrency,max_model_len,threshold=1,recent=None):
    clean=dict(threshold,candidate_mode='current') if isinstance(threshold,dict) else threshold
    result=reference_select(rows,budget,chunk,free,total,block_size,concurrency,max_model_len,clean,recent)
    if not isinstance(threshold,dict):return result
    mode=threshold.get('candidate_mode','current')
    result['features']['candidate_mode']=mode
    args=(free,block_size,concurrency,max_model_len)
    profile=threshold.get('profile_action')
    if profile:
        # TIMING ONLY: no claim that these probes satisfy n or progress guards.
        b,k=profile['budget'],profile['chunk']
        assert 1<=b<=budget and 1<=k<=chunk
        plan=project(rows,b,k,*args)
        assert plan is not None and plan.tokens
        result.update(budget=b,chunk=k,selected=plan,changed=result['base'] is not None and plan.tokens!=result['base'].tokens)
        result['features']['profiling_only']=True
        return result
    if mode=='current':return result
    assert mode in ('points_fixed','points_half','points_spt','points_rate')
    base,n=result['base'],result['threshold']
    if base is None or n is None or base.growth<=0 or len(rows)<2:return result
    table=calibration()
    pairs=[tuple(p) for p in table['pairs']]
    if mode=='points_half':pairs[0],pairs[1]=pairs[1],pairs[0]
    reference=result['selected']
    protected={r.rid:base.tokens[r.rid] for r in rows if r.rid in base.tokens and
               (r.decode or (r.outputs==0 and base.tokens[r.rid]==r.remaining))}
    initial=completion_work(remaining_work(rows,{}))
    diagnostics=[];eligible=[];seen=set()
    for index,(b,k) in enumerate(pairs):
        plan=project(rows,b,k,*args)
        d=dict(budget=b,chunk=k,tokens=None,growth=None,saving=None,guard=False,duplicate=False)
        if plan is not None and plan.tokens:
            signature=tuple(sorted(plan.tokens.items()))
            saving=base.growth-plan.growth
            guard=all(plan.tokens.get(rid,0)>=v for rid,v in protected.items())
            score=completion_work(remaining_work(rows,plan.tokens))
            gain=initial-score
            duration,context,lookup=predicted_ms(rows,plan,k,table)
            d.update(tokens=plan.tokens,growth=plan.growth,saving=saving,guard=guard,
                     duplicate=signature in seen,completion_work=score,work_reduction=gain,
                     predicted_ms=duration,timing_context=context,timing_q=lookup,rate=gain/duration)
            seen.add(signature)
            if guard and saving>=n and not d['duplicate']:
                if mode in ('points_fixed','points_half'):key=(index,)
                elif mode=='points_spt':key=(score,index)
                else:key=(-gain/duration,index)
                eligible.append((key,b,k,plan))
        diagnostics.append(d)
    result.update(budget=budget,chunk=chunk,selected=base,changed=False,candidates=len(pairs),
                  candidate_diagnostics=diagnostics,
                  rejected_infeasible=sum(d['tokens'] is None for d in diagnostics),
                  rejected_guard=sum(d['tokens'] is not None and not d['guard'] for d in diagnostics),
                  rejected_saving=sum(d['guard'] and d['saving']<n for d in diagnostics))
    if eligible:
        _,b,k,plan=min(eligible,key=lambda x:x[0])
        result.update(budget=b,chunk=k,selected=plan,changed=plan.tokens!=base.tokens)
    result['features'].update(reference_saving=base.growth-reference.growth,
        different_from_reference=result['selected'].tokens!=reference.tokens,
        eligible_candidates=len(eligible),initial_completion_work=initial)
    return result
