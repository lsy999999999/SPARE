"""Frozen SPARE selector with a candidate-index switch for leave-one-out tests."""
import math

from research.v1_budget_20260922.policy import project
from research.v1_saving_threshold_20260924.policy import features


def select(rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, threshold=1, recent=None, options=None):
    opt = dict(options or {})
    mode = opt.get('mode', 'spare')
    args = free, block_size, concurrency, max_model_len
    base = project(rows, budget, chunk, *args)
    f = features(rows, base, free, total, block_size, recent)
    work = sum(max(0, r.prompt-r.computed) for r in rows if r.outputs == 0)
    tau = opt.get('tau', threshold.get('tau_s', 1) if isinstance(threshold, dict) else 1)
    ratio = free/(1+(work+tau*f.get('arrival_rate', 0))/block_size)
    n = max(1, min(100000, math.ceil(opt.get('n0', 8)+opt.get('alpha', 128)*ratio)))
    if mode == 'constant':
        n = opt['n']
    f.update(arrival_adjusted_ratio=ratio, arrival_tau_s=tau,
             arrival_expected_tokens=tau*f.get('arrival_rate', 0))
    out = dict(budget=budget, chunk=chunk, base=base, selected=base,
               candidates=0, changed=False, rejected_guard=0, rejected_saving=0,
               rejected_infeasible=0, threshold=n, features=f, candidate_diagnostics=[])
    if mode in ('fixed', 'occupancy'):
        b = opt.get('fixed_budget', 512)
        if mode == 'occupancy' and 1-free/max(1, total-1) < opt.get('occupancy', .7):
            return out
        plan = project(rows, b, min(chunk, b), *args)
        out.update(budget=b, chunk=min(chunk, b), selected=plan, candidates=1,
                   changed=base is not None and (plan is None or plan.tokens != base.tokens))
        return out
    if base is None:
        return out
    protected = {r.rid: base.tokens[r.rid] for r in rows if r.rid in base.tokens and (
        (opt.get('protect_decode', True) and r.decode) or
        (opt.get('protect_finish', True) and r.outputs == 0 and base.tokens[r.rid] == r.remaining))}
    chosen = False
    candidates = ((max(1, budget//4), chunk), (max(1, budget//2), chunk),
                  (budget, min(64, chunk)), (max(1, budget//2), min(64, chunk)))
    axes = opt.get('candidate_axes', 'both')
    if axes == 'budget':
        candidates = candidates[:2]
    elif axes == 'chunk':
        candidates = candidates[2:3]
    elif axes != 'both':
        raise ValueError(axes)
    indices = tuple(opt.get('candidate_indices', range(1, len(candidates)+1)))
    if not indices or len(set(indices)) != len(indices) or any(
            not isinstance(i, int) or not 1 <= i <= len(candidates) for i in indices):
        raise ValueError(indices)
    for i in indices:
        b, c = candidates[i-1]
        plan = project(rows, b, c, *args)
        out['candidates'] += 1
        d = dict(budget=b, chunk=c, growth=None, saving=None, guard=False,
                 deferred_tokens=None, deferred_requests=None, tokens=None)
        if plan is None or not plan.tokens:
            out['rejected_infeasible'] += 1
        else:
            d.update(growth=plan.growth, saving=base.growth-plan.growth,
                     guard=all(plan.tokens.get(rid, 0) >= value for rid, value in protected.items()),
                     deferred_tokens=sum(max(0, v-plan.tokens.get(rid, 0)) for rid, v in base.tokens.items()),
                     deferred_requests=sum(v > plan.tokens.get(rid, 0) for rid, v in base.tokens.items()),
                     tokens=plan.tokens)
            d['acceptance_saving'] = (sum(base.tokens.values()) - sum(plan.tokens.values())) / block_size if opt.get('saving_basis') == 'tokens' else d['saving']
            if not d['guard']:
                out['rejected_guard'] += 1
            elif d['acceptance_saving'] < n:
                out['rejected_saving'] += 1
            elif not chosen and base.growth > 0 and len(rows) >= 2:
                out.update(budget=b, chunk=c, selected=plan, changed=plan.tokens != base.tokens)
                chosen = True
        if opt.get('diagnostics', True):
            out['candidate_diagnostics'].append(d)
        if chosen and opt.get('early_exit', False):
            break
    return out
