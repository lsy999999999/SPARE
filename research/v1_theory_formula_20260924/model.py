"""Conditional TTFT risk-cost model; no claim that runtime parameters are calibrated."""
import math


def benefit(family, free, saving, scale):
    """Integral of demand survival over [F,F+d], in blocks, for continuous demand."""
    assert family in ('lomax','exponential') and free>=0 and saving>=0 and scale>0
    if family=='lomax':
        return scale*math.log1p(saving/(scale+free))
    return scale*math.exp(-free/scale)*(-math.expm1(-saving/scale))


def continuous_threshold(family, free, scale, cost_over_r):
    assert free>=0 and scale>0 and cost_over_r>=0
    if cost_over_r==0:return 0.
    if family=='lomax':
        v=cost_over_r/scale
        log_result=math.log(scale+free)+(v+math.log1p(-math.exp(-v)) if v>1 else math.log(math.expm1(v)))
        return math.exp(log_result) if log_result<700 else math.inf
    assert family=='exponential'
    log_ratio=math.log(cost_over_r/scale)+free/scale
    if log_ratio>=0:return math.inf
    return -scale*math.log1p(-math.exp(log_ratio))


def integer_threshold(family, free, scale, cost_over_r, max_saving):
    n=continuous_threshold(family,free,scale,cost_over_r)
    if not math.isfinite(n) or n>max_saving:return math.inf
    return max(1,math.ceil(n)) if max_saving>=1 else math.inf


def discrete_benefit(free,saving,demand_values,probabilities):
    assert len(demand_values)==len(probabilities)
    assert all(p>=0 for p in probabilities) and math.isclose(sum(probabilities),1.)
    return sum(p*min(saving,max(y-free,0)) for y,p in zip(demand_values,probabilities))
