"""Identical threshold/guards/ordering, only small-chunk scaling varies."""
from research.v1_budget_20260922.policy import project
from research.v1_saving_threshold_20260924.policy import features,threshold_value

def small_chunk(chunk,mode):
    if mode=='fixed64':return min(64,chunk)
    if mode=='ratio16':return max(1,chunk//16)
    if mode=='ratio4':return max(1,chunk//4)
    raise ValueError(mode)

def pairs(budget,chunk,mode):
    k=small_chunk(chunk,mode)
    return [(max(1,budget//4),chunk),(max(1,budget//2),chunk),(budget,k),(max(1,budget//2),k)]

def select(rows,budget,chunk,free,total,block_size,concurrency,max_model_len,threshold=1,recent=None):
    args=(free,block_size,concurrency,max_model_len)
    base=project(rows,budget,chunk,*args)
    extra=dict(recent or {});work=sum(max(0,r.prompt-r.computed) for r in rows if r.outputs==0)
    rate=extra.get('arrival_rate',0.);held=sum(r.blocks for r in rows if r.outputs>0)
    z=held/max(total-1,1);assert 0<=z<=1 and rate>=0
    tau=threshold.get('tau_s',0.) if isinstance(threshold,dict) else 0.
    x=free/(1+(work+tau*rate)/block_size);gate=x/(1+x)
    extra.update(pressure_gate=gate,pressure_decoded_headroom=(1-z)*gate,work_free_ratio=free/(1+work/block_size),
        arrival_adjusted_ratio=x,arrival_expected_tokens=tau*rate,arrival_tau_s=tau,
        decoded_kv_fraction=z,decoded_kv_headroom=1-z,decoded_kv_blocks=held)
    f=features(rows,base,free,total,block_size,extra);n=threshold_value(threshold,f)
    mode=threshold.get('candidate_mode','fixed64') if isinstance(threshold,dict) else 'fixed64'
    f.update(candidate_mode=mode,small_chunk=small_chunk(chunk,mode),selected_candidate=-1)
    result=dict(budget=budget,chunk=chunk,base=base,selected=base,candidates=0,changed=False,
        rejected_guard=0,rejected_saving=0,rejected_infeasible=0,threshold=n,features=f,candidate_diagnostics=[])
    if base is None:return result
    protected={r.rid:base.tokens[r.rid] for r in rows if r.rid in base.tokens and
               (r.decode or (r.outputs==0 and base.tokens[r.rid]==r.remaining))}
    chosen=False
    for index,(b,k) in enumerate(pairs(budget,chunk,mode)):
        plan=project(rows,b,k,*args);result['candidates']+=1
        d=dict(budget=b,chunk=k,growth=None,saving=None,guard=False,tokens=None)
        if plan is None or not plan.tokens:result['rejected_infeasible']+=1
        else:
            d.update(growth=plan.growth,saving=base.growth-plan.growth,tokens=plan.tokens,
                     guard=all(plan.tokens.get(rid,0)>=v for rid,v in protected.items()))
            if not d['guard']:result['rejected_guard']+=1
            elif n is None or d['saving']<n:result['rejected_saving']+=1
            elif not chosen and base.growth>0 and len(rows)>=2:
                result.update(budget=b,chunk=k,selected=plan,changed=plan.tokens!=base.tokens)
                f['selected_candidate']=index;chosen=True
        result['candidate_diagnostics'].append(d)
    return result
