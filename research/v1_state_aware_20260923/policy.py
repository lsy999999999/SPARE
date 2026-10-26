"""Local demand/space gates; no output-length or arrival oracle."""
from fractions import Fraction
from research.v1_budget_20260922.policy import project
from research.v1_capacity_ablation_20260923.policy import select as old_select

NEW = ('pressure_one', 'pressure_relative')


def select(method, rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, occupancy_threshold=.7, pressure='1/4', saving='1/2'):
    if method not in NEW:
        return old_select(method, rows, budget, chunk, free, total, block_size,
                          concurrency, max_model_len, occupancy_threshold)
    p, s = Fraction(pressure), Fraction(saving)
    assert 0 < p <= 1 and 0 < s <= 1
    args = (free, block_size, concurrency, max_model_len)
    base = project(rows, budget, chunk, *args)
    result = dict(budget=budget, chunk=chunk, base=base, selected=base,
                  candidates=0, changed=False, projected=True, rejected_guard=0,
                  rejected_saving=0, rejected_infeasible=0, pressure_triggered=False,
                  candidate_diagnostics=[])
    if base is None or base.growth == 0 or len(rows) < 2:
        return result
    # Integer comparisons preserve exact boundaries; free excludes the null block.
    result['pressure_triggered'] = base.growth*p.denominator >= max(free, 1)*p.numerator
    if not result['pressure_triggered']:
        return result
    threshold = 1 if method == 'pressure_one' else max(
        1, (base.growth*s.numerator+s.denominator-1)//s.denominator)
    protected = {r.rid: base.tokens[r.rid] for r in rows if r.rid in base.tokens
                 and (r.decode or (r.outputs == 0 and base.tokens[r.rid] == r.remaining))}
    for b, c in ((max(1,budget//4),chunk), (max(1,budget//2),chunk),
                 (budget,min(64,chunk)), (max(1,budget//2),min(64,chunk))):
        if (b,c) == (budget,chunk):
            continue
        result['candidates'] += 1
        plan = project(rows,b,c,*args)
        detail = dict(budget=b,chunk=c,growth=None if plan is None else plan.growth,
                      threshold=threshold)
        result['candidate_diagnostics'].append(detail)
        if plan is None or not plan.tokens:
            detail['rejection']='infeasible'; result['rejected_infeasible'] += 1
        elif base.growth-plan.growth < threshold:
            detail['rejection']='saving'; result['rejected_saving'] += 1
        elif any(plan.tokens.get(rid,0)<n for rid,n in protected.items()):
            detail['rejection']='guard'; result['rejected_guard'] += 1
        else:
            detail['rejection']=None
            result.update(budget=b,chunk=c,selected=plan,changed=plan.tokens!=base.tokens)
            break
    return result
