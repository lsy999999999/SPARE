"""Paired-window uncertainty for discrete 2 s capacity differences."""
from __future__ import annotations

import csv

import numpy as np

from research.www_biobjective_ablation_mean_20261019 import run

SEED = 20261019
RESAMPLES = 2000
THETA_MS = 2000.
COMPARE = ('srp', 'no_finish_guard', 'no_decode_guard', 'no_protection',
           'constant16', 'constant32', 'fixed256', 'fixed512',
           'budget_only', 'chunk_only', 'no_arrival', 'token_proxy')


def capacity_bootstrap(source, kv, arm, indices, names):
    group = run.prior.base.read(run.OUT / 'groups' / f'{source}_c{kv}_{arm}.json')
    rates = sorted(group['rates'])
    pass_rows = []
    for rate in rates:
        data = run.prior.base.read(run.rate_path(source, kv, arm, rate))
        records = {r['name']: r for r in data['rows']}
        assert set(records) == set(names)
        ordered = [records[n] for n in names]
        valid = np.array([r['valid'] for r in ordered], dtype=bool)
        ttft = np.array([r.get('mean_ttft_ms', np.inf) for r in ordered], dtype=float)
        tbt = np.array([r.get('p99_tbt_ms', np.inf) for r in ordered], dtype=float)
        drift = np.array([r.get('backlog_drift', np.inf) for r in ordered], dtype=float)
        passed = (valid[indices].all(axis=1)
                  & (ttft[indices].mean(axis=1) <= THETA_MS)
                  & (tbt[indices].mean(axis=1) <= 200.)
                  & (drift[indices].mean(axis=1) <= max(2., rate)))
        pass_rows.append(passed)
    continuous = np.logical_and.accumulate(np.stack(pass_rows, axis=1), axis=1)
    return (continuous * np.asarray(rates)).max(axis=1)


def analyze():
    status = run.prior.base.read(run.OUT / 'status.json')
    assert status['completed'] == status['total'] and not status['errors']
    rng = np.random.default_rng(SEED)
    rows = []
    for source in run.SOURCES:
        names = sorted(s['name'] for s in run.prior.prepare()['samples']
                       if s['source'] == source)
        assert len(names) == 50
        indices = rng.integers(0, len(names), size=(RESAMPLES, len(names)))
        for kv in run.KVS:
            full = capacity_bootstrap(source, kv, 'spare', indices, names)
            full_point = run.prior.base.read(run.OUT / 'groups' /
                                             f'{source}_c{kv}_spare.json')['capacities']['2000']['last_pass_qps']
            for arm in COMPARE:
                other = capacity_bootstrap(source, kv, arm, indices, names)
                point = run.prior.base.read(run.OUT / 'groups' /
                                            f'{source}_c{kv}_{arm}.json')['capacities']['2000']['last_pass_qps']
                delta = other - full
                low, high = np.quantile(delta, (.025, .975))
                rows.append(dict(source=source, kv_blocks=kv, arm=arm,
                                 theta_s=2, capacity_spare_qps=full_point,
                                 capacity_variant_qps=point,
                                 delta_variant_minus_spare_qps=round(point-full_point, 4),
                                 bootstrap_low_qps=round(float(low), 4),
                                 bootstrap_high_qps=round(float(high), 4),
                                 variant_higher_fraction=round(float(np.mean(delta > 0)), 4),
                                 variant_equal_fraction=round(float(np.mean(delta == 0)), 4),
                                 resamples=RESAMPLES, window_unit='same 50 test windows'))
    path = run.OUT / 'analysis' / 'paired_capacity_bootstrap_2s.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == '__main__':
    analyze()
