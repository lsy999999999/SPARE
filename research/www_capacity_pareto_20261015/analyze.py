"""Select configurations on validation streams and compare held-out capacity."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from research.www_capacity_pareto_20261015.run import (
    BUDGETS, CAPACITIES, MAIN, MID_MS, OUT, RATES, SLOS, read, write,
)

WEIGHTS = tuple(round(i / 10, 1) for i in range(11))


def load(split):
    design = read(OUT / 'design.json')
    samples = [s for s in design['samples'] if s['split'] == split]
    configs = {(s['source'], cap): [(arm, b) for arm in MAIN for b in BUDGETS]
               for s in samples for cap in CAPACITIES}
    data = defaultdict(list)
    for s in samples:
        for cap in CAPACITIES:
            for arm, budget in configs[s['source'], cap]:
                path = OUT / split / f"{s['name']}_c{cap}_{arm}_t{budget}.json"
                for row in read(path):
                    data[s['source'], cap, arm, budget, row['rate']].append(row)
    return samples, data


def summarize(rows, slo, rate):
    if not rows:
        raise ValueError('missing rate')
    if any(not r.get('valid', True) for r in rows):
        return dict(mean_ttft_ms=None, p95_ttft_ms=None, p99_tbt_ms=None,
                    mean_backlog_drift=None, drift_limit=None, passed=False,
                    failed_simulations=sum(not r.get('valid', True) for r in rows))
    p95 = float(np.mean([r['p95_ttft_ms'] for r in rows]))
    p99 = float(np.mean([r['p99_tbt_ms'] for r in rows]))
    drift = float(np.mean([r['backlog_drift'] for r in rows]))
    drift_limit = max(2., .05 * rate * (60000. - MID_MS) / 1000.)
    return dict(mean_ttft_ms=float(np.mean([r['mean_ttft_ms'] for r in rows])),
                p95_ttft_ms=p95, p99_tbt_ms=p99, mean_backlog_drift=drift,
                drift_limit=drift_limit,
                passed=p95 <= slo['p95_ttft_ms'] and p99 <= slo['p99_tbt_ms']
                and drift <= drift_limit)


def measure(data, source, cap, arm, budget, slo_name):
    slo = SLOS[slo_name]
    by_rate = {rate: summarize(data[source, cap, arm, budget, rate], slo, rate)
               for rate in RATES}
    passing = [rate for rate, x in by_rate.items() if x['passed']]
    capacity = 0.
    for rate in RATES:
        if rate not in passing:
            break
        capacity = rate
    first_failure = min((r for r in RATES if r > capacity), default=None)
    nonmonotone = any(by_rate[r]['passed'] for r in RATES if r > capacity)
    return dict(source=source, capacity_blocks=cap, arm=arm, budget=budget,
                slo=slo_name, qps=capacity, next_failed_qps=first_failure,
                capacity_lower_bound=first_failure is None, nonmonotone=nonmonotone,
                by_rate=by_rate)


def capacity_ci(data, source, cap, arm, budget, slo_name, rng, draws=500):
    rates = {rate: data[source, cap, arm, budget, rate] for rate in RATES}
    first = rates[RATES[0]]
    by_day = defaultdict(list)
    for index, row in enumerate(first):
        by_day[row['day']].append(index)
    estimates = []
    for _ in range(draws):
        indices = [int(index) for day_indices in by_day.values()
                   for index in rng.choice(day_indices, len(day_indices))]
        passing = [rate for rate in RATES
                   if summarize([rates[rate][i] for i in indices], SLOS[slo_name], rate)['passed']]
        capacity = 0.
        for rate in RATES:
            if rate not in passing:
                break
            capacity = rate
        estimates.append(capacity)
    return [float(x) for x in np.quantile(estimates, (.025, .975))]


def validation():
    status = read(OUT / 'validation_status.json')
    assert status['completed'] == status['expected'] and not status['failures']
    _, data = load('validation')
    points = []
    selection = []
    references = []
    for source in ('code', 'conversation'):
        for cap in CAPACITIES:
            for slo_name in SLOS:
                scene = [measure(data, source, cap, arm, b, slo_name)
                         for arm in MAIN for b in BUDGETS]
                native = max((p for p in scene if p['arm'] == 'native'),
                             key=lambda p: (p['qps'], -p['budget']))
                if native['qps'] == 0:
                    raise RuntimeError(f'no native capacity at {source}/{cap}/{slo_name}')
                ref_rate = max(r for r in RATES if r <= max(RATES[0], native['qps']/2))
                l0 = native['by_rate'][ref_rate]['mean_ttft_ms']
                c0 = native['qps']
                references.append(dict(source=source, capacity=cap, slo=slo_name,
                                       native_budget=native['budget'], ref_rate=ref_rate,
                                       l0_ms=l0, c0_qps=c0))
                for p in scene:
                    p['ref_rate'] = ref_rate
                    p['ref_mean_ttft_ms'] = p['by_rate'][ref_rate]['mean_ttft_ms']
                    p['ref_feasible'] = p['qps'] >= ref_rate
                    points.append(p)
                for arm in MAIN:
                    candidates = [p for p in scene if p['arm'] == arm and p['ref_feasible']]
                    if not candidates:
                        continue
                    for w in WEIGHTS:
                        winner = min(candidates, key=lambda p: (
                            w*p['ref_mean_ttft_ms']/l0 + (1-w)*c0/p['qps'], p['budget']))
                        selection.append(dict(source=source, capacity=cap,
                                              slo=slo_name, arm=arm, weight=w,
                                              budget=winner['budget'],
                                              validation_qps=winner['qps'],
                                              validation_ttft_ms=winner['ref_mean_ttft_ms']))
    write(OUT / 'validation_points.json', points)
    write(OUT / 'selection.json', dict(configs=selection, references=references,
                                      weights=WEIGHTS, rates=RATES))
    print(f'{len(points)} validation configurations; {len(selection)} selections')


def test():
    status = read(OUT / 'test_status.json')
    assert status['completed'] == status['expected'] and not status['failures']
    _, data = load('test')
    frozen = read(OUT / 'selection.json')
    refs = {(r['source'], r['capacity'], r['slo']): r for r in frozen['references']}
    measured = {}
    rng = np.random.default_rng(20261015)
    for source in ('code', 'conversation'):
        for cap in CAPACITIES:
            for slo in SLOS:
                ref_rate = refs[source, cap, slo]['ref_rate']
                for arm in MAIN:
                    for budget in BUDGETS:
                        p = measure(data, source, cap, arm, budget, slo)
                        p['ref_rate'] = ref_rate
                        p['ref_mean_ttft_ms'] = p['by_rate'][ref_rate]['mean_ttft_ms']
                        p['ref_feasible'] = p['qps'] >= ref_rate
                        measured[source, cap, slo, arm, budget] = p
    rows = []
    for r in frozen['configs']:
        p = measured[r['source'], r['capacity'], r['slo'], r['arm'], r['budget']]
        if 'capacity_ci_95' not in p:
            p['capacity_ci_95'] = capacity_ci(data, r['source'], r['capacity'],
                                              r['arm'], r['budget'], r['slo'], rng)
        rows.append(dict(**r, test_qps=p['qps'], next_failed_qps=p['next_failed_qps'],
                         capacity_ci_95=p['capacity_ci_95'],
                         capacity_lower_bound=p['capacity_lower_bound'],
                         nonmonotone=p['nonmonotone'],
                         ref_rate=p['ref_rate'],
                         test_ttft_ms=p['ref_mean_ttft_ms'],
                         test_ref_feasible=p['ref_feasible'],
                         test_p95_ttft_ms=p['by_rate'][p['ref_rate']]['p95_ttft_ms'],
                         test_p99_tbt_ms=p['by_rate'][p['ref_rate']]['p99_tbt_ms']))
    write(OUT / 'test_points.json', list(measured.values()))
    write(OUT / 'selected_test.json', rows)
    checks = []
    for source in ('code', 'conversation'):
        for cap in CAPACITIES:
            for slo in SLOS:
                for w in WEIGHTS:
                    scene = [r for r in rows if (r['source'], r['capacity'], r['slo'], r['weight'])
                             == (source, cap, slo, w) and r['test_ref_feasible']]
                    ours = next((r for r in scene if r['arm'] == 'arrival'), None)
                    if ours is None:
                        checks.append(dict(source=source, capacity=cap, slo=slo, weight=w,
                                           spare_feasible=False, dominated=None, dominators=[]))
                        continue
                    competitors = [p for (s, c, z, arm, _), p in measured.items()
                                   if (s, c, z) == (source, cap, slo)
                                   and arm != 'arrival' and p['ref_feasible']]
                    dominators = [p for p in competitors
                                  if p['ref_mean_ttft_ms'] <= ours['test_ttft_ms']
                                  and p['qps'] >= ours['test_qps']
                                  and (p['ref_mean_ttft_ms'] < ours['test_ttft_ms'] or
                                       p['qps'] > ours['test_qps'])]
                    checks.append(dict(source=source, capacity=cap, slo=slo, weight=w,
                                       spare_feasible=True, dominated=bool(dominators),
                                       dominators=[f"{p['arm']}:T{p['budget']}" for p in dominators]))
    write(OUT / 'dominance.json', checks)
    report(rows, checks, refs)


def report(rows, checks, refs):
    lines = ['# Mean TTFT 与 Serving Capacity：CPU simulator', '',
             'Capacity 是满足 P95 TTFT、P99 TBT 和队列积压条件的最大通过到达率，'
             '按离散 QPS 网格给出通过下界及下一失败上界。mean TTFT 在验证集冻结的统一到达率下比较。'
             '测试为每类负载50个独立窗口，来自两个日期；结果是冻结时钟与方法适配器上的模拟结果。', '',
             '|负载|KV块|SLO|参考QPS|SPARE非支配权重|SPARE选中容量QPS|',
             '|---|---:|---|---:|---:|---|']
    for source in ('code', 'conversation'):
        for cap in CAPACITIES:
            for slo in SLOS:
                scene = [r for r in checks if (r['source'], r['capacity'], r['slo'])
                         == (source, cap, slo)]
                ours = [r for r in rows if (r['source'], r['capacity'], r['slo'], r['arm'])
                        == (source, cap, slo, 'arrival')]
                qps = sorted({r['test_qps'] for r in ours})
                count = sum(r['spare_feasible'] and not r['dominated'] for r in scene)
                lines.append(f"|{source}|{cap}|{slo}|{refs[source, cap, slo]['ref_rate']:.1f}|"
                             f"{count}/11|{', '.join(f'{q:g}' for q in qps)}|")
    lines += ['', '逐权重配置与测试结果见 selected_test.json；完整 QPS 曲线及通过判据见 '
              'test_points.json，支配关系见 dominance.json。容量到达扫描使用泊松流，'
              '每窗预热20秒、测量40秒；请求结束后的排空不计入容量。'
              '各策略共享请求、到达种子和同一拟合时钟。', '',
              '外部方法是共同 V1 后端上的规则／代理实现。'
              '本轮未运行 GPU 推理，也未重新验证时钟模型在这些持续负载上的预测精度。']
    (OUT / 'REPORT_CN.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('validation', 'test'))
    args = parser.parse_args()
    validation() if args.stage == 'validation' else test()
