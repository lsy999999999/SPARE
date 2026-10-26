"""Frozen A plus the predeclared occupancy and ablation controls."""
from research.v1_budget_20260922.policy import choose as frozen_choose, project

METHODS = ('native', 'srp_matched', 'a_frozen', 'a_shadow', 'fixed_small',
           'occupancy', 'occupancy_finish_guard', 'a_no_finish_guard',
           'a_any_positive_saving')


def select(method, rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, occupancy_threshold=.9):
    args = (free, block_size, concurrency, max_model_len)
    result = dict(budget=budget, chunk=chunk, base=None, selected=None,
                  candidates=0, changed=False, projected=False,
                  rejected_guard=0, rejected_saving=0, rejected_infeasible=0)
    if method in ('native', 'srp_matched', 'fixed_small'):
        return result
    if method in ('a_frozen', 'a_shadow'):
        result.update(frozen_choose(rows, budget, chunk, free, total, block_size,
                                   concurrency, max_model_len), projected=True)
        return result
    if method == 'occupancy':
        # No projection/search in the simple policy's decision path.
        high = 1-free/(total-1) >= occupancy_threshold
        result.update(budget=512 if high else budget, pressure_triggered=high)
        return result
    base = project(rows, budget, chunk, *args)
    result.update(base=base, selected=base, projected=True)
    if base is None:
        return result
    protected = {r.rid: base.tokens[r.rid] for r in rows if r.rid in base.tokens
                 and (r.decode or (method != 'a_no_finish_guard' and r.outputs == 0
                                   and base.tokens[r.rid] == r.remaining))}
    if method == 'occupancy_finish_guard':
        if 1-free/(total-1) < occupancy_threshold:
            return result
        candidates = [(512, chunk)]
        threshold = 0
    else:
        threshold = 1 if method == 'a_any_positive_saving' else (total*7+255)//256
        if base.growth < threshold or len(rows) < 2:
            return result
        candidates = [(max(1, budget//4), chunk), (max(1, budget//2), chunk),
                      (budget, min(64, chunk)), (max(1, budget//2), min(64, chunk))]
    for b, c in candidates:
        if (b, c) == (budget, chunk):
            continue
        result['candidates'] += 1
        plan = project(rows, b, c, *args)
        if plan is None or not plan.tokens:
            result['rejected_infeasible'] += 1
            continue
        if base.growth-plan.growth < threshold:
            result['rejected_saving'] += 1
            continue
        if any(plan.tokens.get(rid, 0) < n for rid, n in protected.items()):
            result['rejected_guard'] += 1
            continue
        result.update(budget=b, chunk=c, selected=plan, changed=plan.tokens != base.tokens)
        break
    return result
