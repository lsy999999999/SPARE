"""Small observable-state ablations. No future arrivals or output targets."""
from dataclasses import replace
from math import ceil

from research.v1_budget_20260922.policy import project
from research.v1_state_aware_20260923.policy import select as old_select

METHODS = ('a_any_positive_saving', 'fixed_small', 'pressure_one',
           'pressure_early', 'pressure_min_delay', 'horizon4_first',
           'horizon8_first', 'horizon4_min_delay', 'horizon4_risk', 'horizon8_risk')


def advance_rows(rows, plan, block_size):
    """Conditional next observable state: every completed step emits one token."""
    result = []
    for row in rows:
        n = plan.tokens.get(row.rid, 0)
        if not n:
            result.append(row)
            continue
        computed = row.computed + n
        emitted = int(computed == row.tokens)
        result.append(replace(row, computed=computed, tokens=row.tokens + emitted,
                              outputs=row.outputs + emitted, running=True,
                              blocks=max(row.blocks, ceil(computed / block_size))))
    # Stable ties preserve the original queue's arrival/id tie ordering.
    running = [r for r in result if r.running]
    return ([r for r in running if r.decode]
            + sorted((r for r in running if not r.decode), key=lambda r: r.remaining)
            + sorted((r for r in result if not r.running), key=lambda r: r.remaining))


def forecast(rows, first, horizon, budget, chunk, block_size, concurrency, max_model_len):
    """Potential growth with no completions; first action only, then shared base."""
    growth = 0
    trajectory = []
    plan = first
    for k in range(horizon):
        if k:
            plan = project(rows, budget, chunk, 10**12, block_size, concurrency, max_model_len)
        growth += plan.growth
        trajectory.append(growth)
        rows = advance_rows(rows, plan, block_size)
    return trajectory


def select(method, rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, occupancy_threshold=.7, pressure='1/4', saving='1/2'):
    args = (budget, chunk, free, total, block_size, concurrency, max_model_len)
    if method in ('a_any_positive_saving', 'fixed_small', 'pressure_one'):
        return old_select(method, rows, *args, occupancy_threshold, pressure, saving)
    if method == 'pressure_early':
        return old_select('pressure_one', rows, *args, occupancy_threshold, '1/8', saving)
    assert method in METHODS, method
    base = project(rows, budget, chunk, free, block_size, concurrency, max_model_len)
    result = dict(budget=budget, chunk=chunk, base=base, selected=base,
                  candidates=0, changed=False, projected=True, rejected_guard=0,
                  rejected_saving=0, rejected_infeasible=0, pressure_triggered=False,
                  candidate_diagnostics=[], probe={})
    if base is None or len(rows) < 2:
        return result
    horizon = 8 if method.startswith('horizon8') else 4
    fargs = (horizon, budget, chunk, block_size, concurrency, max_model_len)
    if method.startswith('horizon'):
        trajectory = forecast(rows, base, *fargs)
        risk = max(0, trajectory[-1] - free)
        triggered = risk > 0
        result['probe'].update(horizon=horizon, base_trajectory=trajectory, base_risk=risk)
    else:
        risk = None
        triggered = base.growth > 0 and base.growth * 4 >= max(free, 1)
    result['pressure_triggered'] = triggered
    if not triggered or base.growth == 0:
        return result
    protected = {r.rid: base.tokens[r.rid] for r in rows if r.rid in base.tokens
                 and (r.decode or (r.outputs == 0 and base.tokens[r.rid] == r.remaining))}
    eligible = []
    candidates = ((max(1, budget//4), chunk), (max(1, budget//2), chunk),
                  (budget, min(64, chunk)), (max(1, budget//2), min(64, chunk)))
    for index, (b, c) in enumerate(candidates):
        if (b, c) == (budget, chunk):
            continue
        plan = project(rows, b, c, free, block_size, concurrency, max_model_len)
        result['candidates'] += 1
        detail = dict(budget=b, chunk=c, growth=None if plan is None else plan.growth,
                      threshold=1, rejection=None)
        result['candidate_diagnostics'].append(detail)
        if plan is None or not plan.tokens:
            detail['rejection'] = 'infeasible'
        elif base.growth - plan.growth < 1:
            detail['rejection'] = 'saving'
        elif any(plan.tokens.get(rid, 0) < n for rid, n in protected.items()):
            detail['rejection'] = 'guard'
        if detail['rejection']:
            result['rejected_' + detail['rejection']] += 1
            continue
        deferred = [max(0, base.tokens.get(r.rid, 0)-plan.tokens.get(r.rid, 0))
                    for r in rows if r.outputs == 0]
        detail.update(delta_g=base.growth-plan.growth, deferred_tokens=sum(deferred),
                      deferred_requests=sum(n > 0 for n in deferred))
        if method.endswith('_risk'):
            trajectory = forecast(rows, plan, *fargs)
            detail.update(trajectory=trajectory, risk=max(0, trajectory[-1]-free))
            rank = (detail['risk'], sum(deferred), detail['deferred_requests'], index)
        else:
            rank = (sum(deferred), detail['deferred_requests'], -detail['delta_g'], index)
        eligible.append((rank, b, c, plan, detail))
        if method.endswith('_first'):
            break
    if not eligible:
        return result
    chosen = min(eligible, key=lambda x: x[0])
    _, b, c, plan, detail = chosen
    if method.endswith('_risk') and detail['risk'] >= risk:
        result['probe']['risk_rejected_all'] = True
        return result
    result.update(budget=b, chunk=c, selected=plan, changed=plan.tokens != base.tokens)
    result['probe']['chosen'] = detail
    return result
