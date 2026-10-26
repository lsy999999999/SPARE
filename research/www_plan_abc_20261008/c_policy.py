"""Independently calibrated fixed candidate with G2 feasibility and guards."""
from research.v1_budget_20260922.policy import project
from research.www_evaluation_gpu_completion_20261003.policy import select as spare_select


def select(rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, threshold=1, recent=None, options=None):
    options = dict(options or {})
    if options.get("mode") != "fixed_guarded":
        return spare_select(rows, budget, chunk, free, total, block_size,
                            concurrency, max_model_len, threshold, recent, options)
    fixed_budget = options["fixed_budget"]
    choice = spare_select(rows, budget, chunk, free, total, block_size,
                          concurrency, max_model_len, threshold, recent, {})
    base = choice["base"]
    choice.update(budget=budget, chunk=chunk, selected=base, changed=False,
                  candidates=0, rejected_guard=0, rejected_saving=0,
                  rejected_infeasible=0, candidate_diagnostics=[])
    if base is None or base.growth <= 0 or len(rows) < 2:
        return choice
    candidate_chunk = min(chunk, fixed_budget)
    plan = project(rows, fixed_budget, candidate_chunk, free, block_size,
                   concurrency, max_model_len)
    choice["candidates"] = 1
    if plan is None or not plan.tokens:
        choice["rejected_infeasible"] = 1
        return choice
    protected = {r.rid: base.tokens[r.rid] for r in rows if r.rid in base.tokens
                 and (r.decode or (r.outputs == 0 and base.tokens[r.rid] == r.remaining))}
    guard = all(plan.tokens.get(rid, 0) >= amount for rid, amount in protected.items())
    choice["candidate_diagnostics"] = [dict(budget=fixed_budget, chunk=candidate_chunk,
        growth=plan.growth, saving=base.growth-plan.growth, guard=guard,
        tokens=plan.tokens)]
    if not guard:
        choice["rejected_guard"] = 1
        return choice
    choice.update(budget=fixed_budget, chunk=candidate_chunk,
                  selected=plan, changed=plan.tokens != base.tokens)
    return choice
