"""Request/arrival-only deferral selector for a KV-blind control."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from research.v1_budget_20260922.policy import Plan, project


@dataclass(frozen=True)
class RequestView:
    rid: str
    computed: int
    tokens: int
    prompt: int
    outputs: int
    running: bool

    @property
    def remaining(self):
        return self.tokens - self.computed

    @property
    def decode(self):
        return self.outputs > 0 and self.remaining == 1


def request_views(rows):
    return [RequestView(row.rid, row.computed, row.tokens, row.prompt,
                        row.outputs, row.running) for row in rows]


def token_plan(rows, budget, chunk, max_running, max_model_len):
    tokens = {}
    running = sum(row.running for row in rows)
    for row in rows:
        if budget <= 0 or (not row.running and running >= max_running):
            break
        count = min(row.remaining, budget, chunk if chunk > 0 else row.remaining)
        if row.running:
            count = min(count, max_model_len - 1 - row.computed)
        if count <= 0:
            continue
        tokens[row.rid] = count
        budget -= count
        if not row.running:
            running += 1
    return Plan(tokens, 0)


def decide(rows, budget, chunk, max_running, max_model_len, arrival_rate):
    """Return a budget/chunk action without reading KV allocations or free space."""
    base = token_plan(rows, budget, chunk, max_running, max_model_len)
    threshold_tokens = max(1, chunk - ceil(max(0.0, arrival_rate)))
    result = dict(budget=budget, chunk=chunk, base=base,
                  selected=base, candidates=0, changed=False,
                  rejected_guard=0, rejected_saving=0,
                  rejected_infeasible=0, threshold_tokens=threshold_tokens)
    if len(rows) < 2 or not base.tokens:
        return result
    protected = {row.rid: base.tokens[row.rid] for row in rows
                 if row.rid in base.tokens and (row.decode or
                     (row.outputs == 0 and base.tokens[row.rid] == row.remaining))}
    candidates = ((max(1, budget // 4), chunk),
                  (max(1, budget // 2), chunk),
                  (budget, min(64, chunk)),
                  (max(1, budget // 2), min(64, chunk)))
    for candidate_budget, candidate_chunk in candidates:
        plan = token_plan(rows, candidate_budget, candidate_chunk,
                          max_running, max_model_len)
        result["candidates"] += 1
        if not plan.tokens:
            result["rejected_infeasible"] += 1
        elif any(plan.tokens.get(rid, 0) < amount
                 for rid, amount in protected.items()):
            result["rejected_guard"] += 1
        elif sum(base.tokens.values()) - sum(plan.tokens.values()) < threshold_tokens:
            result["rejected_saving"] += 1
        else:
            result.update(budget=candidate_budget, chunk=candidate_chunk,
                          selected=plan, changed=plan.tokens != base.tokens)
            break
    return result


def select(rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, threshold=1, recent=None, options=None):
    assert (options or {}).get("mode") == "kv_blind"
    arrival_rate = max(0.0, (recent or {}).get("arrival_rate", 0.0))
    action = decide(request_views(rows), budget, chunk, concurrency, max_model_len,
                    arrival_rate)
    # This physical projection is only for the simulator's execution audit.
    physical_base = project(rows, budget, chunk, free,
                            block_size, concurrency, max_model_len)
    executed = project(rows, action["budget"], action["chunk"], free,
                       block_size, concurrency, max_model_len)
    return dict(action, base=physical_base, selected=executed,
                threshold=action["threshold_tokens"],
                features=dict(arrival_rate=arrival_rate,
                              threshold_tokens=action["threshold_tokens"],
                              planned_deferred_tokens=sum(action["base"].tokens.values())
                              - sum(action["selected"].tokens.values())))
