"""Same arrival threshold; candidate-specific progress loss and inverse block budgets."""
import math
from research.v1_broad12h_20260925.policy import select as original_select
from research.v1_budget_20260922.policy import project


def old_pairs(budget, chunk):
    return [(max(1, budget // 4), chunk), (max(1, budget // 2), chunk),
            (budget, min(64, chunk)), (max(1, budget // 2), min(64, chunk))]


def progress_loss(rows, base, plan, block):
    # Lost increase in -log(remaining + B), counted only for requests yet to emit.
    # This is a dimensionless proxy, NOT a measured waiting-time cost.
    return sum(max(0., math.log((max(0, r.prompt-r.computed-plan.tokens.get(r.rid, 0))+block) /
                               (max(0, r.prompt-r.computed-base.tokens.get(r.rid, 0))+block)))
               for r in rows if r.outputs == 0)


def inverse_budget(rows, chunk, target_growth, budget, block, concurrency, max_len):
    # Invert unconstrained required growth, which is monotone in total budget.
    # Actual project(free) is NOT generally monotone: waiting admission can stop.
    low, high = 0, budget
    while low < high:
        mid = (low + high + 1) // 2
        p = project(rows, mid, chunk, 10**12, block, concurrency, max_len)
        assert p is not None
        if p.growth <= target_growth:
            low = mid
        else:
            high = mid - 1
    return low


def select(rows, budget, chunk, free, total, block_size, concurrency, max_model_len,
           threshold=1, recent=None):
    result = original_select(rows, budget, chunk, free, total, block_size,
                             concurrency, max_model_len, threshold, recent)
    mode = threshold.get('candidate_mode', 'current') if isinstance(threshold, dict) else 'current'
    forced = threshold.get('forced_action') if isinstance(threshold, dict) else None
    base, n = result['base'], result['threshold']
    result['features']['candidate_mode'] = mode
    if base is None:
        assert forced is None, 'Cannot force an action in recovery'
        return result
    args = (free, block_size, concurrency, max_model_len)
    protected = {r.rid: base.tokens[r.rid] for r in rows if r.rid in base.tokens
                 and (r.decode or (r.outputs == 0 and base.tokens[r.rid] == r.remaining))}
    if forced is not None:
        b, k = forced['budget'], forced['chunk']
        assert 1 <= b <= budget and 1 <= k <= chunk
        plan = project(rows, b, k, *args)
        assert plan is not None and plan.tokens
        assert all(plan.tokens.get(rid, 0) >= count for rid, count in protected.items())
        assert plan.tokens == base.tokens or (n is not None and base.growth-plan.growth >= n)
        result.update(budget=b, chunk=k, selected=plan, changed=plan.tokens != base.tokens)
        result['features']['forced_action'] = forced
        return result
    if mode == 'current' or n is None or base.growth <= 0 or len(rows) < 2:
        return result
    assert mode in ('rank4', 'grid', 'dynamic')
    pairs = old_pairs(budget, chunk)
    dynamic_pairs = []
    if mode == 'grid':
        pairs += [(max(1, b), k) for b in (budget//4, budget//2, 3*budget//4, budget)
                  for k in sorted({min(64, chunk), min(128, chunk), chunk})
                  if (max(1, b), k) != (budget, chunk)]
    elif mode == 'dynamic' and base.growth >= n:
        active = [r for r in rows if base.tokens.get(r.rid, 0) > 0]
        slack = sum(max(0, r.blocks*block_size-r.computed) for r in active)
        protected_chunk = max(protected.values(), default=1)
        for saving in sorted({n, min(2*n, base.growth)}):
            target = base.growth-saving
            fair_chunk = block_size * ((block_size*target+slack)//(max(1, len(active))*block_size))
            memory_chunk = min(chunk, max(min(block_size, chunk), protected_chunk, fair_chunk))
            for k in sorted({chunk, memory_chunk}):
                b = inverse_budget(rows, k, target, budget, block_size, concurrency, max_model_len)
                if b > 0:
                    dynamic_pairs.append((b, k))
        pairs += dynamic_pairs
    diagnostics, eligible, seen, seen_pairs = [], [], set(), set()
    for b, k in pairs:
        if (b, k) in seen_pairs:
            continue
        seen_pairs.add((b, k))
        plan = project(rows, b, k, *args)
        row = dict(budget=b, chunk=k, growth=None, saving=None, guard=False,
                   tokens=None, loss=None, loss_per_block=None, duplicate=False,
                   dynamic=(b, k) in dynamic_pairs)
        if plan is not None and plan.tokens:
            saving = base.growth-plan.growth
            guard = all(plan.tokens.get(rid, 0) >= count for rid, count in protected.items())
            key = tuple(sorted(plan.tokens.items()))
            duplicate = key in seen
            seen.add(key)
            loss = progress_loss(rows, base, plan, block_size)
            row.update(growth=plan.growth, saving=saving, guard=guard, tokens=plan.tokens,
                       loss=loss, loss_per_block=loss/saving if saving > 0 else None,
                       duplicate=duplicate)
            if guard and saving >= n and not duplicate:
                eligible.append(((loss/saving, loss, -saving, -sum(plan.tokens.values()), len(diagnostics)), b, k, plan))
        diagnostics.append(row)
    result.update(budget=budget, chunk=chunk, selected=base, changed=False,
                  candidates=len(diagnostics), candidate_diagnostics=diagnostics,
                  rejected_infeasible=sum(d['tokens'] is None for d in diagnostics),
                  rejected_guard=sum(d['tokens'] is not None and not d['guard'] for d in diagnostics),
                  rejected_saving=sum(d['guard'] and d['saving'] < n for d in diagnostics))
    result['features'].update(dynamic_pairs=[list(p) for p in dynamic_pairs],
                              unique_candidate_plans=len(seen), eligible_candidates=len(eligible))
    if eligible:
        score, b, k, plan = min(eligible, key=lambda x: x[0])
        result.update(budget=b, chunk=k, selected=plan, changed=plan.tokens != base.tokens)
        result['features'].update(selected_loss_per_block=score[0], selected_progress_loss=score[1])
    return result
