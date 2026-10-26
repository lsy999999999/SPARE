"""Paper rules on observable request state; no realized output-length inputs."""
from bisect import bisect_right
from collections import deque
from math import ceil, exp, expm1, floor, log
import random


def future_peak_blocks(states, block_size=16):
    """Past-Future Eq. 2-4, evaluated before each predicted completion.

    Each tuple is (prompt, already_generated, predicted_total_output).
    Rounding to physical blocks is the explicit V1 adaptation.
    """
    if not states:
        return 0
    if any(p < 1 or g < 0 or length <= g for p, g, length in states):
        raise ValueError('Prediction must exceed observed generation progress')
    return max(sum(ceil((p + g + t) / block_size)
                   for p, g, length in states if length - g >= t)
               for t in {length - g for _, g, length in states})


class HistoryPredictor:
    """Algorithm 1: one conditional empirical draw per request per round."""
    def __init__(self, history=(), seed=0):
        if any(int(x) <= 0 for x in history):
            raise ValueError('History lengths must be positive')
        self.history = deque(map(int, history), maxlen=1000)
        self.rng = random.Random(seed)
        self._sorted = None

    def observe(self, completed_output):
        if completed_output > 0:
            self.history.append(int(completed_output))
            self._sorted = None

    def sample(self, generated, common_cap):
        if common_cap <= generated:
            raise ValueError('No legal next output token')
        if self._sorted is None:
            self._sorted = sorted(self.history)
        first = bisect_right(self._sorted, generated)
        value = (self._sorted[self.rng.randrange(first, len(self._sorted))]
                 if first < len(self._sorted) else common_cap)
        return max(generated + 1, min(value, common_cap))


def geometric_quantize(value, bin_size):
    """UniBoost Eq. 8: geometric MemGuard bins anchored at ``bin_size``."""
    if value < 0 or bin_size <= 0:
        raise ValueError('value must be nonnegative and bin_size positive')
    return bin_size * 2 ** floor(log(max(value, bin_size) / bin_size, 2))


def gamma_boost(value, gamma):
    """UniBoost Eq. 2 evaluated stably for positive attained service."""
    if value <= 0 or gamma <= 0:
        raise ValueError('value and gamma must be positive')
    return -log(-expm1(-gamma * value)) / gamma


def uniboost_priority(arrival_time, signal, gamma=1e-4, bin_size=16):
    return arrival_time - gamma_boost(geometric_quantize(signal, bin_size), gamma)


def apt_kv_density(pending_time, required_blocks, slo_s=1.0):
    """Apt-Serve Eq. 7 KV-only (beta=0) greedy value density.

    The paper demotes an already violated request to a near-zero value so it
    cannot create a chain of further SLO violations.
    """
    if required_blocks <= 0 or pending_time < 0 or slo_s <= 0:
        raise ValueError('invalid Apt-Serve density input')
    value = pending_time if pending_time <= slo_s else 1e-12
    return value / required_blocks


def censored_log_t_score(mu, sigma, cap, beta, *, alpha=.9, nu=3.5,
                         samples=2048, seed=0):
    """TIE Eq. 9-11 Monte-Carlo score with deterministic common samples."""
    if sigma <= 0 or cap <= 0 or not 0 < alpha < 1 or beta < 0:
        raise ValueError('invalid TIE distribution parameters')
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        normal = rng.gauss(0., 1.)
        chi_square = rng.gammavariate(nu / 2, 2.)
        student = normal / ((chi_square / nu) ** .5)
        values.append(min(cap, exp(min(log(cap), mu + sigma * student))))
    values.sort()
    tail = values[int(alpha * len(values)):]
    return sum(values) / len(values) + beta * sum(tail) / len(tail)
