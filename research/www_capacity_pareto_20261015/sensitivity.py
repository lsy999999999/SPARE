"""Post-hoc SLO sensitivity using the completed CPU-simulator trajectories."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.www_capacity_pareto_20261015.run import (
    BUDGETS, CAPACITIES, MAIN, OUT, RATES, read, write,
)

WEIGHTS = tuple(round(i / 10, 1) for i in range(11))
SOURCES = ('code', 'conversation')


def load_points(split):
    return {(p['source'], p['capacity_blocks'], p['arm'], p['budget']): p
            for p in read(OUT / f'{split}_points.json') if p['slo'] == 'strict'}


def capacity(point, p95_limit, tbt_limit, drift_fraction, drift_floor):
    passed = 0.
    for rate in RATES:
        row = point['by_rate'][str(rate)]
        drift_limit = max(drift_floor, drift_fraction * rate * 20.)
        if (row['p95_ttft_ms'] is None or row['p95_ttft_ms'] > p95_limit
                or row['p99_tbt_ms'] > tbt_limit
                or row['mean_backlog_drift'] > drift_limit):
            break
        passed = rate
    return passed


def evaluate_scene(validation, test, source, blocks, p95_limit, tbt_limit,
                   drift_fraction, drift_floor):
    keys = [(source, blocks, arm, budget) for arm in MAIN for budget in BUDGETS]
    val_caps = {key: capacity(validation[key], p95_limit, tbt_limit,
                              drift_fraction, drift_floor) for key in keys}
    native = max((key for key in keys if key[2] == 'native'),
                 key=lambda key: (val_caps[key], -key[3]))
    if val_caps[native] == 0:
        return dict(source=source, blocks=blocks, valid=False,
                    reason='native has no passing rate')
    ref_rate = max(rate for rate in RATES if rate <= max(RATES[0], val_caps[native] / 2))
    l0 = validation[native]['by_rate'][str(ref_rate)]['mean_ttft_ms']
    c0 = val_caps[native]
    val_ttft = {key: validation[key]['by_rate'][str(ref_rate)]['mean_ttft_ms']
                for key in keys}
    test_caps = {key: capacity(test[key], p95_limit, tbt_limit,
                               drift_fraction, drift_floor) for key in keys}
    test_ttft = {key: test[key]['by_rate'][str(ref_rate)]['mean_ttft_ms']
                 for key in keys}
    competitors = [key for key in keys if key[2] != 'arrival'
                   and test_caps[key] >= ref_rate]
    selected = []
    for weight in WEIGHTS:
        options = [key for key in keys if key[2] == 'arrival'
                   and val_caps[key] >= ref_rate]
        if not options:
            selected.append(dict(weight=weight, feasible=False, dominated=None))
            continue
        ours = min(options, key=lambda key: (
            weight * val_ttft[key] / l0 + (1 - weight) * c0 / val_caps[key], key[3]))
        feasible = test_caps[ours] >= ref_rate
        dominators = []
        if feasible:
            dominators = [key for key in competitors
                          if test_caps[key] >= test_caps[ours]
                          and test_ttft[key] <= test_ttft[ours]
                          and (test_caps[key] > test_caps[ours]
                               or test_ttft[key] < test_ttft[ours])]
        selected.append(dict(weight=weight, budget=ours[3], feasible=feasible,
                             dominated=bool(dominators) if feasible else None,
                             qps=test_caps[ours], ttft_ms=test_ttft[ours],
                             dominators=[f'{key[2]}:T{key[3]}' for key in dominators]))
    return dict(source=source, blocks=blocks, valid=True, ref_rate=ref_rate,
                non_dominated=sum(row['feasible'] and not row['dominated']
                                  for row in selected), selected=selected)


def scan(p95_values, tbt_values, drift_fraction, drift_floor):
    validation, test = load_points('validation'), load_points('test')
    rows = []
    for tbt_limit in tbt_values:
        for p95_limit in p95_values:
            scenes = [evaluate_scene(validation, test, source, blocks, p95_limit,
                                     tbt_limit, drift_fraction, drift_floor)
                      for source in SOURCES for blocks in CAPACITIES]
            rows.append(dict(p95_ttft_ms=p95_limit, p99_tbt_ms=tbt_limit,
                             drift_fraction=drift_fraction, drift_floor=drift_floor,
                             scenes=scenes,
                             scenes_with_spare_point=sum(s.get('non_dominated', 0) > 0
                                                         for s in scenes),
                             non_dominated_weights=sum(s.get('non_dominated', 0)
                                                       for s in scenes)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p95-min', type=int, default=1000)
    parser.add_argument('--p95-max', type=int, default=15000)
    parser.add_argument('--p95-step', type=int, default=250)
    parser.add_argument('--tbt', default='50,100,200,500')
    parser.add_argument('--drift-fraction', type=float, default=.05)
    parser.add_argument('--drift-floor', type=float, default=2.)
    parser.add_argument('--output', type=Path, default=OUT / 'slo_sensitivity.json')
    args = parser.parse_args()
    rows = scan(range(args.p95_min, args.p95_max + 1, args.p95_step),
                [float(x) for x in args.tbt.split(',')],
                args.drift_fraction, args.drift_floor)
    write(args.output, rows)
    best = sorted(rows, key=lambda row: (row['scenes_with_spare_point'],
                                         row['non_dominated_weights']), reverse=True)
    for row in best[:12]:
        print(f"P95={row['p95_ttft_ms']}ms TBT={row['p99_tbt_ms']:g}ms "
              f"scenes={row['scenes_with_spare_point']}/4 "
              f"weights={row['non_dominated_weights']}/44")
    print(f'{len(rows)} settings; all-scene settings: '
          f"{sum(row['scenes_with_spare_point'] == 4 for row in rows)}")


if __name__ == '__main__':
    main()
