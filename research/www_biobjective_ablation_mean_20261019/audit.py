"""Check trace identity for protection variants on matched archived rates."""
from __future__ import annotations

import json

from research.www_biobjective_ablation_mean_20261019 import run


def compare(left, right):
    pairs = mismatches = 0
    by_condition = []
    for source in run.SOURCES:
        for kv in run.KVS:
            a = run.prior.base.read(run.OUT / 'groups' / f'{source}_c{kv}_{left}.json')
            b = run.prior.base.read(run.OUT / 'groups' / f'{source}_c{kv}_{right}.json')
            rates = sorted(set(a['rates']) & set(b['rates']))
            count = changed = 0
            for rate in rates:
                left_rows = run.prior.base.read(run.rate_path(source, kv, left, rate))['rows']
                right_rows = run.prior.base.read(run.rate_path(source, kv, right, rate))['rows']
                x = {r['name']: r['trajectory_sha256'] for r in left_rows if r['valid']}
                y = {r['name']: r['trajectory_sha256'] for r in right_rows if r['valid']}
                assert x.keys() == y.keys()
                count += len(x)
                changed += sum(x[name] != y[name] for name in x)
            pairs += count
            mismatches += changed
            by_condition.append(dict(source=source, kv_blocks=kv,
                                     common_rates=len(rates), pairs=count,
                                     different_trajectories=changed))
    return dict(left=left, right=right, total_pairs=pairs,
                different_trajectories=mismatches, by_condition=by_condition)


def audit():
    run.prepare()
    results = [compare('spare', 'no_decode_guard'),
               compare('no_finish_guard', 'no_protection'),
               compare('spare', 'no_finish_guard')]
    assert results[0]['different_trajectories'] == 0
    assert results[1]['different_trajectories'] == 0
    run.prior.base.write(run.OUT / 'analysis' / 'trajectory_audit.json', results)
    print(json.dumps([{k: r[k] for k in ('left', 'right', 'total_pairs',
                                        'different_trajectories')} for r in results], indent=2))


if __name__ == '__main__':
    audit()
