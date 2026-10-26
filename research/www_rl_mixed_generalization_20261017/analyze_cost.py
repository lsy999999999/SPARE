"""Summarize paired scheduling-time measurements on one CPU host."""
from __future__ import annotations

import json

import numpy as np

from research.www_rl_mixed_generalization_20261017 import cost_mixed, run


OUT = run.OUT / 'cost_same_host_20261018_analysis.json'


def main():
    raw = run.old.base.read(cost_mixed.OUT)
    assert raw['stage'] == 'complete' and len(raw['rows']) == 100
    rng = np.random.default_rng(1018)
    results = {}
    for source in ('code', 'conversation', 'all'):
        chosen = [row for row in raw['rows']
                  if source == 'all' or row['source'] == source]
        spare = np.array([row['spare']['schedule_mean_ms'] for row in chosen])
        rl = np.array([row['rl_mixed']['schedule_mean_ms'] for row in chosen])
        draws = rng.integers(0, len(chosen), (10_000, len(chosen)))
        ratios = rl[draws].mean(axis=1) / spare[draws].mean(axis=1)
        results[source] = dict(windows=len(chosen),
                               spare_mean_schedule_ms=float(spare.mean()),
                               rl_mean_schedule_ms=float(rl.mean()),
                               rl_over_spare=float(rl.mean() / spare.mean()),
                               paired_bootstrap_ratio_95=[float(v) for v in
                                                          np.quantile(ratios, (.025, .975))])
    payload = dict(metric=raw['metric'], kv_blocks=raw['kv_blocks'],
                   rate_qps=raw['rate_qps'], rl_seed=raw['rl_seed'],
                   bootstrap_seed=1018, bootstrap_draws=10_000,
                   inference_scope='paired test windows, not repeated host timing runs',
                   results=results)
    run.old.base.write(OUT, payload)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
