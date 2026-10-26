"""Fit a nonnegative service curve on prior current r0; diagnose on prior r1."""
import gzip
import itertools
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from research.v1_completion_order_20260927.common import *


def main():
    assert not (OUT/'service_model.json').exists()
    previous = ROOT/'artifacts/v1_dynamic_actions_20260927_r1'
    samples = {0: [], 1: []}
    provenance = {}
    for path in sorted((previous/'groups').glob('*/runs/*_current_r*.json')):
        result = read(path)
        if result['phase'] != 'formal':
            continue
        provenance[str(path.relative_to(ROOT))] = digest(path)
        with gzip.open(previous/result['rounds_path'], 'rt') as f:
            rows = json.load(f)
        for row in rows:
            if row['preemptions']:
                continue
            tokens = row['scheduled']
            attention = sum(tokens.get(r['rid'],0)*r['computed']+
                            tokens.get(r['rid'],0)*(tokens.get(r['rid'],0)+1)/2 for r in row['state'])
            samples[result['repeat']].append(([1, sum(tokens.values()), attention/1e6],
                                              row['step_ms']-row['schedule_ms']))
    x = np.array([a for a, b in samples[0]])
    y = np.array([b for a, b in samples[0]])
    assert min(y) > 0
    choices = []
    # Exactly solve three-variable nonnegative least squares by all active sets.
    for bits in itertools.product((False, True), repeat=3):
        if not any(bits):
            continue
        ids = np.flatnonzero(bits)
        beta = np.linalg.lstsq(x[:, ids], y, rcond=None)[0]
        if min(beta) < 0:
            continue
        coef = np.zeros(3); coef[ids] = beta
        choices.append((float(np.sum((x@coef-y)**2)), coef))
    _, coef = min(choices, key=lambda pair:pair[0])
    diagnostics = {}
    for repeat, values in samples.items():
        a = np.array([x for x,y in values]); b = np.array([y for x,y in values])
        predicted = a@coef
        relative = np.abs(predicted-b)/b
        diagnostics[str(repeat)] = dict(rounds=len(b), mean_absolute_error_ms=float(np.mean(np.abs(predicted-b))),
            median_relative_error=float(np.median(relative)), p95_relative_error=float(np.quantile(relative,.95)))
    write(OUT/'service_model.json', dict(coefficients=coef.tolist(),
        expression='ms = intercept + token_coefficient * tokens + attention_coefficient * attention_pairs / 1e6',
        fit='NNLS, prior current repeat 0 normal rounds; all samples, no tail trimming',
        validation='prior current repeat 1; same windows, not independent workload generalization',
        diagnostics=diagnostics, source_result_hashes=provenance))
    print(read(OUT/'service_model.json')['coefficients'], diagnostics)


if __name__ == '__main__':
    main()
