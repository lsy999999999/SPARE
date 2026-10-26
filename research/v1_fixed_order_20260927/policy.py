"""Separate fixed-order changes from candidate extent and reference saving."""
from research.v1_dynamic_actions_20260927.policy import select as legacy_select, old_pairs, inverse_budget
from research.v1_completion_order_20260927.policy import completion_work, remaining_work
from research.v1_budget_20260922.policy import project


def select(rows, budget, chunk, free, total, block_size, concurrency, max_model_len,
           threshold=1, recent=None):
    if not isinstance(threshold, dict):
        return legacy_select(rows, budget, chunk, free, total, block_size, concurrency,
                             max_model_len, threshold, recent)
    mode=threshold.get('candidate_mode','current')
    result=legacy_select(rows,budget,chunk,free,total,block_size,concurrency,max_model_len,
                         dict(threshold,candidate_mode='current'),recent)
    result['features']['candidate_mode']=mode
    if mode=='current' or threshold.get('forced_action') is not None:
        return result
    assert mode in ('half_first','spt4','dyn_spt','dyn_save','dyn_pareto')
    base,n,reference=result['base'],result['threshold'],result['selected']
    if base is None or n is None or base.growth<=0 or len(rows)<2:
        return result
    ref_budget,ref_chunk=result['budget'],result['chunk']
    ref_saving=base.growth-reference.growth
    pending=[r for r in rows if r.outputs==0]
    ref_progress=sum(reference.tokens.get(r.rid,0) for r in pending)
    base_progress=sum(base.tokens.get(r.rid,0) for r in pending)
    ref_score=completion_work(remaining_work(rows,reference.tokens))
    protected={r.rid:base.tokens[r.rid] for r in rows if r.rid in base.tokens and
               (r.decode or (r.outputs==0 and base.tokens[r.rid]==r.remaining))}
    args=(free,block_size,concurrency,max_model_len)
    pairs=old_pairs(budget,chunk)
    if mode=='half_first':pairs=[pairs[1],pairs[0],pairs[2],pairs[3]]
    dynamic_pairs=[]
    if mode.startswith('dyn_') and base.growth>=n:
        active=[r for r in rows if base.tokens.get(r.rid,0)>0]
        slack=sum(max(0,r.blocks*block_size-r.computed) for r in active)
        protected_chunk=max(protected.values(),default=1)
        # All three dynamic arms see exactly the same proposals. Adding the
        # reference target enables a real comparison at reference saving.
        targets={n,min(2*n,base.growth)}
        if ref_saving>0:targets.add(ref_saving)
        for saving in sorted(targets):
            target=base.growth-saving
            fair=block_size*((block_size*target+slack)//(max(1,len(active))*block_size))
            memory_chunk=min(chunk,max(min(block_size,chunk),protected_chunk,fair))
            for k in sorted({chunk,memory_chunk}):
                b=inverse_budget(rows,k,target,budget,block_size,concurrency,max_model_len)
                if b>0:dynamic_pairs.append((b,k))
        pairs+=dynamic_pairs
    required=max(n,ref_saving) if mode in ('dyn_save','dyn_pareto') else n
    diagnostics=[];eligible=[];seen=set();seen_pairs=set()
    for b,k in pairs:
        if (b,k) in seen_pairs:continue
        seen_pairs.add((b,k))
        plan=project(rows,b,k,*args)
        d=dict(budget=b,chunk=k,growth=None,saving=None,tokens=None,guard=False,duplicate=False,
               progress_guard=False,reference_guard=False,score=None,dynamic=(b,k) in dynamic_pairs)
        if plan is not None and plan.tokens:
            saving=base.growth-plan.growth
            guard=all(plan.tokens.get(rid,0)>=v for rid,v in protected.items())
            signature=tuple(sorted(plan.tokens.items()))
            duplicate=signature in seen;seen.add(signature)
            progress=sum(plan.tokens.get(r.rid,0) for r in pending)
            score=completion_work(remaining_work(rows,plan.tokens))
            # Fixed-set arms change order only: no additional progress guard.
            progress_guard=not (mode.startswith('dyn_') and base_progress>0 and progress==0)
            ref_guard=(progress>=ref_progress and score<=ref_score and
                all(plan.tokens.get(r.rid,0)>=reference.tokens.get(r.rid,0) for r in rows if r.outputs>0))
            d.update(growth=plan.growth,saving=saving,tokens=plan.tokens,guard=guard,duplicate=duplicate,
                     progress_guard=progress_guard,reference_guard=ref_guard,score=score,prefill_progress=progress)
            valid=guard and saving>=required and not duplicate and progress_guard
            if mode=='dyn_pareto':
                # Require strict improvement in at least one monitored metric;
                # otherwise retain current, including its legitimate fallback.
                strict=(saving>ref_saving or progress>ref_progress or score<ref_score or
                        any(plan.tokens.get(r.rid,0)>reference.tokens.get(r.rid,0) for r in rows if r.outputs>0))
                valid=valid and ref_guard and strict
            if valid:
                key=(len(diagnostics),) if mode=='half_first' else (score,-saving,-sum(plan.tokens.values()),len(diagnostics))
                eligible.append((key,b,k,plan))
        diagnostics.append(d)
    if mode!='dyn_pareto':
        result.update(budget=budget,chunk=chunk,selected=base,changed=False)
    result.update(candidates=len(diagnostics),candidate_diagnostics=diagnostics,
        rejected_infeasible=sum(d['tokens'] is None for d in diagnostics),
        rejected_guard=sum(d['tokens'] is not None and not d['guard'] for d in diagnostics),
        rejected_saving=sum(d['guard'] and d['saving']<required for d in diagnostics))
    result['features'].update(reference_budget=ref_budget,reference_chunk=ref_chunk,
        reference_saving=ref_saving,reference_prefill_progress=ref_progress,reference_completion_work=ref_score,
        required_saving=required,base_prefill_progress=base_progress,dynamic_pairs=[list(p) for p in dynamic_pairs],
        unique_candidate_plans=len(seen),eligible_candidates=len(eligible))
    if eligible:
        key,b,k,plan=min(eligible,key=lambda item:item[0])
        result.update(budget=b,chunk=k,selected=plan,changed=plan.tokens!=base.tokens)
        result['features']['selected_order_score']=key[0]
    result['features']['different_from_reference']=result['selected'].tokens!=reference.tokens
    return result
