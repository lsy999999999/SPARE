"""Generate proposals without using measured branch outcomes or output lengths."""
from research.v1_candidate_audit_20260928.common import signature, spec
from research.v1_dynamic_actions_20260927.policy import select, old_pairs, inverse_budget
from research.v1_budget_20260922.policy import State, project


def enumerate_actions(row, budget=1024, chunk=256):
    rows = [State(**r) for r in row['state']]
    args = (row['free_before'], 16, 32, 8176)
    ref = select(rows, budget, chunk, row['free_before'], row['total_blocks'],
                 16, 32, 8176, spec(), {'arrival_rate':row['features']['arrival_rate']})
    base, n = ref['base'], ref['threshold']
    if base is None or n is None:
        return dict(actions=[], proposals=[], recovery=True)
    protected = {r.rid:base.tokens[r.rid] for r in rows if r.rid in base.tokens and
                 (r.decode or (r.outputs==0 and base.tokens[r.rid]==r.remaining))}
    pairs = {}
    def add(b, k, family):
        if 1 <= b <= budget and 1 <= k <= chunk:
            pairs.setdefault((b,k), []).append(family)
    for b,k in old_pairs(budget,chunk): add(b,k,'old')
    add(budget,chunk,'base')
    add(ref['budget'],ref['chunk'],'current')
    # Crossed proportions plus absolute 64: these differ at base K=128.
    ks = sorted({max(1,chunk//4), max(1,chunk//2), chunk, min(64,chunk)})
    for k in ks:
        for b in (budget//4,budget//2,3*budget//4,budget): add(b,k,'grid')
    active = [r for r in rows if base.tokens.get(r.rid,0)>0]
    slack = sum(max(0,r.blocks*16-r.computed) for r in active)
    targets = {n, min(2*n,base.growth), base.growth-ref['selected'].growth}
    for saving in sorted(targets):
        if saving <= 0 or saving > base.growth: continue
        growth = base.growth-saving
        fair = 16*((16*growth+slack)//(max(1,len(active))*16))
        memory_k = min(chunk,max(min(16,chunk),max(protected.values(),default=1),fair))
        for k in sorted({chunk,memory_k}):
            add(inverse_budget(rows,k,growth,budget,16,32,8176),k,'inverse')
    # At most two nearest pending completion boundaries; no output-end oracle.
    boundaries = sorted({r.remaining for r in rows if r.outputs==0 and 0<r.remaining<=chunk})[:2]
    for k in boundaries:
        for b in (budget//2,budget): add(b,k,'completion')
    proposals=[]; unique={}
    for (b,k),families in pairs.items():
        plan=project(rows,b,k,*args)
        guard=plan is not None and bool(plan.tokens) and all(plan.tokens.get(rid,0)>=v for rid,v in protected.items())
        saving=None if plan is None else base.growth-plan.growth
        legal=bool(guard and (plan.tokens==base.tokens or saving>=n))
        p=dict(budget=b,chunk=k,families=families,legal=legal,guard=bool(guard),saving=saving,
               tokens=None if plan is None else plan.tokens,growth=None if plan is None else plan.growth)
        proposals.append(p)
        if not legal: continue
        sig=signature(plan.tokens)
        if sig not in unique:
            unique[sig]=dict(id=f'a{len(unique):02d}',budget=b,chunk=k,tokens=plan.tokens,growth=plan.growth,
                saving=saving,aliases=[],families=[],q=sum(plan.tokens.values()),
                prefill=sum(plan.tokens.get(r.rid,0) for r in rows if r.outputs==0),
                completions=sum(r.outputs==0 and 0<r.remaining==plan.tokens.get(r.rid,0) for r in rows),
                partial=sum(r.outputs==0 and 0<r.computed+plan.tokens.get(r.rid,0)<r.prompt for r in rows),
                scheduled_requests=len(plan.tokens),free_after=row['free_before']-plan.growth)
        a=unique[sig]; a['aliases'].append([b,k]);a['families']=sorted(set(a['families']+families))
    actions=list(unique.values())
    legal_ps=[p for p in proposals if p['legal']]
    k_pairs=[]
    for i,a in enumerate(legal_ps):
        for b in legal_ps[i+1:]:
            if a['budget']==b['budget'] and a['chunk']!=b['chunk'] and a['tokens']!=b['tokens']:
                k_pairs.append(dict(budget=a['budget'],chunks=[a['chunk'],b['chunk']],
                    equal_q=sum(a['tokens'].values())==sum(b['tokens'].values()),equal_growth=a['growth']==b['growth']))
    old_sigs={signature(p['tokens']) for p in legal_ps if 'old' in p['families'] or 'base' in p['families']}
    return dict(actions=actions,proposals=proposals,recovery=False,threshold=n,
                k_effective=bool(k_pairs),k_matched=any(p['equal_q'] and p['equal_growth'] for p in k_pairs),
                k_pairs=k_pairs,old_plans=len(old_sigs),new_plans=sum(signature(a['tokens']) not in old_sigs for a in actions),
                current_id=next(a['id'] for a in actions if a['tokens']==ref['selected'].tokens))
