"""Frozen structural candidates; risk distributions are hypotheses, not measurements."""
import math
from research.v1_saving_threshold_20260924.policy import select as base_select


def values(free, work, held, total, block_size=16):
    q = 1 + work / block_size
    x = free / q
    z = held / max(total - 1, 1)
    linear = 8 + 128 * x
    delta = 128 / q
    m = q / 16
    lomax = min(100000., linear * math.expm1(delta) / delta)
    log_required = math.log(delta) + free / m
    exponential = 100000. if log_required >= 0 else min(100000., -m * math.log1p(-math.exp(log_required)))
    g = 8 + 32 * (1-z)
    coupled = min(100000., g * (1+16*x))
    log_expm1 = g/m + math.log1p(-math.exp(-g/m)) if g/m > 1 else math.log(math.expm1(g/m))
    coupled_exact = math.exp(min(math.log(100000.), math.log(m+free)+log_expm1))
    return dict(work_free_ratio=x, decoded_kv_headroom=1-z,
                decoded_kv_fraction=z, decoded_kv_blocks=held,
                lomax_threshold=lomax, exponential_threshold=exponential,
                coupled_decode_threshold=coupled, coupled_exact_threshold=coupled_exact)


def select(rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, threshold=1, recent=None):
    work = sum(max(0, r.prompt-r.computed) for r in rows if r.outputs == 0)
    held = sum(r.blocks for r in rows if r.outputs > 0)
    extra = dict(recent or {})
    # Same feature computation cost for every arm, including controls.
    extra.update(values(free, work, held, total, block_size))
    return base_select(rows, budget, chunk, free, total, block_size, concurrency,
                       max_model_len, threshold, extra)


def arms():
    def spec(a, names, weights):
        return dict(kind='linear', intercept=a, features=names, coefficients=weights)
    return dict(old=spec(8,['work_free_ratio'],[128]),
                decode=spec(8,['work_free_ratio','decoded_kv_headroom'],[128,32]),
                constant16=spec(24,['work_free_ratio'],[128]),
                lomax=spec(0,['lomax_threshold'],[1]),
                exponential=spec(0,['exponential_threshold'],[1]),
                coupled_decode=spec(0,['coupled_decode_threshold'],[1]),
                coupled_exact=spec(0,['coupled_exact_threshold'],[1]))
