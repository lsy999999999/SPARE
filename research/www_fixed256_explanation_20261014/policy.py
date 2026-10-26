"""Decision variants for the fixed-256 explanation experiment."""
import math

from research.v1_budget_20260922.policy import project
from research.v1_saving_threshold_20260924.policy import features


def select(rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, threshold=1, recent=None, options=None):
    opt = dict(options or {})
    mode = opt.get("mode", "full")
    if mode not in {"full", "fixed", "accept256", "recovery256", "no_finish"}:
        raise ValueError(mode)
    args = free, block_size, concurrency, max_model_len
    base = project(rows, budget, chunk, *args)
    f = features(rows, base, free, total, block_size, recent)
    work = sum(max(0, r.prompt - r.computed) for r in rows if r.outputs == 0)
    tau = threshold.get("tau_s", 1) if isinstance(threshold, dict) else 1
    ratio = free / (1 + (work + tau * f.get("arrival_rate", 0)) / block_size)
    n = max(1, min(100000, math.ceil(8 + 128 * ratio)))
    f.update(arrival_adjusted_ratio=ratio, arrival_tau_s=tau,
             arrival_expected_tokens=tau * f.get("arrival_rate", 0))
    out = dict(budget=budget, chunk=chunk, base=base, selected=base,
               candidates=0, changed=False, rejected_guard=0,
               rejected_saving=0, rejected_infeasible=0, threshold=n,
               features=f, candidate_diagnostics=[], variant_triggered=False,
               decision_reason="default")

    if mode == "fixed":
        plan = project(rows, 256, min(chunk, 256), *args)
        out.update(budget=256, chunk=min(chunk, 256), selected=plan,
                   candidates=1, changed=base is not None and
                   (plan is None or plan.tokens != base.tokens),
                   variant_triggered=True, decision_reason="fixed256")
        return out
    if base is None:
        if mode == "recovery256":
            plan = project(rows, 256, min(chunk, 256), *args)
            out.update(budget=256, chunk=min(chunk, 256), selected=plan,
                       candidates=1, variant_triggered=True,
                       decision_reason="recovery256")
        else:
            out["decision_reason"] = "base_recovery"
        return out

    protected = {r.rid: base.tokens[r.rid] for r in rows
                 if r.rid in base.tokens and
                 (r.decode or (mode != "no_finish" and r.outputs == 0 and
                               base.tokens[r.rid] == r.remaining))}
    candidates = ((max(1, budget // 4), chunk),
                  (max(1, budget // 2), chunk),
                  (budget, min(64, chunk)),
                  (max(1, budget // 2), min(64, chunk)))
    chosen = False
    for index, (b, c) in enumerate(candidates):
        plan = project(rows, b, c, *args)
        out["candidates"] += 1
        d = dict(budget=b, chunk=c, growth=None, saving=None, guard=False,
                 deferred_tokens=None, deferred_requests=None, tokens=None)
        if plan is None or not plan.tokens:
            out["rejected_infeasible"] += 1
        else:
            d.update(growth=plan.growth, saving=base.growth - plan.growth,
                     guard=all(plan.tokens.get(rid, 0) >= value
                               for rid, value in protected.items()),
                     deferred_tokens=sum(max(0, v - plan.tokens.get(rid, 0))
                                         for rid, v in base.tokens.items()),
                     deferred_requests=sum(v > plan.tokens.get(rid, 0)
                                           for rid, v in base.tokens.items()),
                     tokens=plan.tokens)
            d["acceptance_saving"] = d["saving"]
            if not d["guard"]:
                out["rejected_guard"] += 1
            elif d["saving"] < n and not (mode == "accept256" and index == 0):
                out["rejected_saving"] += 1
            elif not chosen and base.growth > 0 and len(rows) >= 2:
                out.update(budget=b, chunk=c, selected=plan,
                           changed=plan.tokens != base.tokens)
                chosen = True
                if mode == "accept256" and index == 0 and d["saving"] < n:
                    out.update(variant_triggered=True,
                               decision_reason="accept256_below_threshold")
                elif mode == "no_finish" and index >= 0:
                    out["decision_reason"] = "candidate_after_guard_change"
                else:
                    out["decision_reason"] = "candidate_accepted"
        out["candidate_diagnostics"].append(d)
    return out
