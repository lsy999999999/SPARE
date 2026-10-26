"""Exact physical-saving thresholds; features contain no future information."""
import math
from research.v1_budget_20260922.policy import project


def features(rows, base, free, total, block_size, recent=None):
    pre = [r for r in rows if r.outputs == 0]
    dec = [r for r in rows if r.decode]
    g = base.growth if base is not None else 0
    remaining = [max(0, r.prompt - r.computed) for r in pre]
    occupied = sum(r.blocks for r in rows)
    result = dict(capacity=total - 1, free=free, free_fraction=free / max(total - 1, 1),
        growth=g, pressure=g / max(free, 1), runway=free / max(g, 1),
        pending_prefill=len(pre), remaining_prefill=sum(remaining),
        mean_remaining=sum(remaining) / max(len(pre), 1),
        min_remaining=min(remaining, default=0),
        partial_prefill=sum(0 < r.computed < r.prompt for r in pre),
        decode_count=len(dec), running_count=sum(r.running for r in rows),
        queued_count=sum(not r.running for r in rows),
        decode_context=sum(r.computed for r in dec),
        computed_prefix=sum(r.computed for r in rows),
        occupied_blocks=occupied,
        decode_boundary=sum(r.computed + 1 > r.blocks * block_size for r in dec),
        block_slack=sum(max(0, r.blocks * block_size - r.computed) for r in rows if r.running),
        recomputing=sum(r.outputs > 0 and not r.decode for r in rows),
        completing_prefill=sum(base is not None and base.tokens.get(r.rid, 0) == r.remaining
                              and r.remaining > 0 for r in pre),
        remaining_over_free=sum(remaining) / (block_size * max(free, 1)),
        decode_fraction=len(dec) / max(len(rows), 1))
    result.update(recent or {})
    return result


def threshold_value(spec, f):
    if spec == 'off':
        return None
    if isinstance(spec, (int, float)):
        assert spec >= 1
        return int(spec)
    kind = spec['kind']
    if kind == 'constant':
        return threshold_value(spec['n'], f)
    if kind == 'tree':
        node = spec['tree']
        while 'feature' in node:
            node = node['left' if f[node['feature']] <= node['cut'] else 'right']
        return threshold_value(node['n'], f)
    if kind in ('linear', 'loglinear'):
        z = spec['intercept']
        for feature, coefficient in zip(spec['features'], spec['coefficients']):
            x = f[feature]
            if kind == 'loglinear':
                x = math.log1p(max(x, 0))
            z += coefficient * x
        if kind == 'loglinear':
            z = math.expm1(max(-20, min(20, z)))
        return max(1, min(100000, math.ceil(z)))
    raise ValueError(kind)


def select(rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, threshold=1, recent=None):
    args = (free, block_size, concurrency, max_model_len)
    base = project(rows, budget, chunk, *args)
    f = features(rows, base, free, total, block_size, recent)
    n = threshold_value(threshold, f)
    result = dict(budget=budget, chunk=chunk, base=base, selected=base,
        candidates=0, changed=False, rejected_guard=0, rejected_saving=0,
        rejected_infeasible=0, threshold=n, features=f, candidate_diagnostics=[])
    if base is None:
        return result
    protected = {r.rid: base.tokens[r.rid] for r in rows if r.rid in base.tokens
                 and (r.decode or (r.outputs == 0 and base.tokens[r.rid] == r.remaining))}
    chosen = False
    # Evaluate all four uniformly, including off, for unbiased diagnostic coverage.
    for b, c in ((max(1, budget // 4), chunk), (max(1, budget // 2), chunk),
                 (budget, min(64, chunk)), (max(1, budget // 2), min(64, chunk))):
        plan = project(rows, b, c, *args)
        result['candidates'] += 1
        d = dict(budget=b, chunk=c, growth=None, saving=None, guard=False,
                 deferred_tokens=None, deferred_requests=None, tokens=None)
        if plan is None or not plan.tokens:
            result['rejected_infeasible'] += 1
        else:
            d.update(growth=plan.growth, saving=base.growth - plan.growth,
                     guard=all(plan.tokens.get(rid, 0) >= count for rid, count in protected.items()),
                     deferred_tokens=sum(max(0, count - plan.tokens.get(rid, 0)) for rid, count in base.tokens.items()),
                     deferred_requests=sum(count > plan.tokens.get(rid, 0) for rid, count in base.tokens.items()),
                     tokens=plan.tokens)
            if not d['guard']:
                result['rejected_guard'] += 1
            elif n is None or d['saving'] < n:
                result['rejected_saving'] += 1
            elif not chosen and base.growth > 0 and len(rows) >= 2:
                result.update(budget=b, chunk=c, selected=plan, changed=plan.tokens != base.tokens)
                chosen = True
        result['candidate_diagnostics'].append(d)
    return result
