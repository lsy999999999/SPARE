"""Section 12: observable work-normalized free KV, optionally length CV."""
import math
from research.v1_saving_threshold_20260924.policy import select as old_select


def observable(rows, free, block_size):
    remaining = [max(0, r.prompt-r.computed) for r in rows if r.outputs == 0]
    work = sum(remaining)
    mean = work / max(1, len(remaining))
    variance = sum((x-mean)**2 for x in remaining) / max(1, len(remaining))
    return dict(work_free_ratio=free/(1+work/block_size),
                remaining_cv=math.sqrt(variance)/mean if mean else 0.)


def select(rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, threshold=1, recent=None):
    extra = dict(recent or {})
    extra.update(observable(rows, free, block_size))
    return old_select(rows, budget, chunk, free, total, block_size, concurrency,
                      max_model_len, threshold, extra)


def formula(a, b, k=0):
    assert b >= 0
    return dict(kind='linear', intercept=a,
                features=['work_free_ratio', 'remaining_cv'], coefficients=[b, k])
