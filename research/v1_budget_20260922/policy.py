"""Allocation arithmetic for vLLM 0.9.2 V1, single full-attention KV group.

No model execution, scheduler copies, future arrivals or output-length oracle.
This is a NEW candidate, not an action-equivalent port of frozen A v1.
"""
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class State:
    rid: str
    computed: int
    tokens: int
    prompt: int
    outputs: int
    blocks: int
    running: bool

    @property
    def remaining(self):
        return self.tokens - self.computed

    @property
    def decode(self):
        return self.outputs > 0 and self.remaining == 1


@dataclass
class Plan:
    tokens: dict
    growth: int


def project(rows, budget, chunk, free, block_size, max_running, max_model_len):
    """Exact normal branch; None means native running-request preemption needed.

Rows contain running requests first, followed by waiting requests. Each list
uses the actual queue order. APC/speculation/sliding-window/encoder excluded.
"""
    alloc = {}
    growth = 0
    running = sum(r.running for r in rows)
    for r in rows:
        if budget <= 0:
            break
        if not r.running and running >= max_running:
            break
        n = min(r.remaining, budget, chunk if chunk > 0 else r.remaining)
        if r.running:
            n = min(n, max_model_len - 1 - r.computed)
        if n <= 0:
            continue
        extra = max(0, ceil((r.computed + n) / block_size) - r.blocks)
        if extra > free:
            if r.running:
                return None
            break
        free -= extra
        growth += extra
        budget -= n
        alloc[r.rid] = n
        if not r.running:
            running += 1
    return Plan(alloc, growth)


def choose(rows, budget, chunk, free, total, block_size, max_running, max_model_len):
    args = (free, block_size, max_running, max_model_len)
    base = project(rows, budget, chunk, *args)
    result = dict(budget=budget, chunk=chunk, base=base, selected=base,
                  candidates=0, changed=False, fallback=base is None)
    # Historical 0.02734375 growth-fraction threshold = 7/256. It is fixed,
    # not tuned on this matrix. Integer block arithmetic avoids float drift.
    threshold = (total * 7 + 255) // 256
    if base is None or base.growth < threshold or len(rows) < 2:
        return result
    protected = {r.rid: base.tokens[r.rid] for r in rows if r.rid in base.tokens
                 and (r.decode or (r.outputs == 0 and base.tokens[r.rid] == r.remaining))}
    for b, c in [(max(1, budget//4), chunk), (max(1, budget//2), chunk),
                 (budget, min(64, chunk)), (budget, chunk),
                 (max(1, budget//2), min(64, chunk))]:
        if (b, c) == (budget, chunk):
            continue
        result['candidates'] += 1
        plan = project(rows, b, c, *args)
        if plan is None or not plan.tokens or base.growth - plan.growth < threshold:
            continue
        if any(plan.tokens.get(rid, 0) < n for rid, n in protected.items()):
            continue
        result.update(budget=b, chunk=c, selected=plan,
                      changed=plan.tokens != base.tokens)
        break
    return result
