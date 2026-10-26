"""Summarize the 24-window confirmation with paired uncertainty."""
from __future__ import annotations

import csv

import numpy as np

from research.www_component_ablation_mean_20261019 import confirm

RESAMPLES = 2000
SEED = 20261020


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def records(source, kv, arm, rate):
    rows = confirm.pilot.old.prior.base.read(
        confirm.rate_path(source, kv, arm, rate))['rows']
    assert len(rows) == 24
    return {row['name']: row for row in rows}


def capacity_bootstrap(source, kv, arm, names, indices):
    group = confirm.pilot.old.prior.base.read(confirm.OUT / 'groups' /
            f'{source}_c{kv}_{arm}.json')
    rates = sorted(group['rates'])
    passing = []
    for rate in rates:
        rows = records(source, kv, arm, rate)
        ordered = [rows[name] for name in names]
        valid = np.array([row['valid'] for row in ordered], dtype=bool)
        ttft = np.array([row.get('mean_ttft_ms', np.inf) for row in ordered], dtype=float)
        tbt = np.array([row.get('p99_tbt_ms', np.inf) for row in ordered], dtype=float)
        drift = np.array([row.get('backlog_drift', np.inf) for row in ordered], dtype=float)
        passing.append(valid[indices].all(axis=1)
                       & (ttft[indices].mean(axis=1) <= 2000)
                       & (tbt[indices].mean(axis=1) <= 200)
                       & (drift[indices].mean(axis=1) <= max(2, rate)))
    continuous = np.logical_and.accumulate(np.stack(passing, axis=1), axis=1)
    return (continuous * np.asarray(rates)).max(axis=1)


def analyze():
    status = confirm.pilot.old.prior.base.read(confirm.OUT / 'status.json')
    assert status['completed'] == status['total'] and not status['errors']
    samples = confirm.prepare()
    rng = np.random.default_rng(SEED)
    points, comparisons, boots = [], [], []
    for source in confirm.pilot.SOURCES:
        names = sorted(s['name'] for s in samples if s['source'] == source)
        indices = rng.integers(0, 24, size=(RESAMPLES, 24))
        for kv in confirm.pilot.KVS:
            groups = {arm: confirm.pilot.old.prior.base.read(confirm.OUT /
                      'groups' / f'{source}_c{kv}_{arm}.json') for arm in confirm.ARMS}
            full_boot = capacity_bootstrap(source, kv, 'spare', names, indices)
            for arm in confirm.ARMS:
                for theta in confirm.pilot.THETA_MS:
                    cap = groups[arm]['capacities'][f'{theta:g}']
                    points.append(dict(source=source, kv_blocks=kv, arm=arm,
                                       theta_s=theta/1000,
                                       capacity_qps=cap['last_pass_qps'],
                                       next_fail_qps=cap['first_fail_qps'],
                                       failure_causes=';'.join(cap['failure_causes'])))
                    if arm == 'spare':
                        continue
                    full_cap = groups['spare']['capacities'][f'{theta:g}']['last_pass_qps']
                    comparisons.append(dict(source=source, kv_blocks=kv, arm=arm,
                                            theta_s=theta/1000,
                                            delta_capacity_vs_full_qps=round(
                                                cap['last_pass_qps']-full_cap, 4)))
                if arm == 'spare':
                    continue
                common = set(groups['spare']['rates']) & set(groups[arm]['rates'])
                changed = count = 0
                for rate in common:
                    full = records(source, kv, 'spare', rate)
                    other = records(source, kv, arm, rate)
                    assert full.keys() == other.keys()
                    changed += sum(full[name]['trajectory_sha256'] != other[name]['trajectory_sha256']
                                   for name in names if full[name]['valid'] and other[name]['valid'])
                    count += sum(full[name]['valid'] and other[name]['valid'] for name in names)
                full = records(source, kv, 'spare', 2.)
                other = records(source, kv, arm, 2.)
                mean_delta = (sum(other[name]['mean_ttft_ms']-full[name]['mean_ttft_ms']
                                  for name in names)/24 if all(full[name]['valid'] and
                                  other[name]['valid'] for name in names) else None)
                variant_boot = capacity_bootstrap(source, kv, arm, names, indices)
                delta = variant_boot-full_boot
                low, high = np.quantile(delta, (.025, .975))
                boots.append(dict(source=source, kv_blocks=kv, arm=arm, theta_s=2,
                                  delta_capacity_qps=round(
                                      groups[arm]['capacities']['2000']['last_pass_qps']-
                                      groups['spare']['capacities']['2000']['last_pass_qps'], 4),
                                  bootstrap_low_qps=round(float(low), 4),
                                  bootstrap_high_qps=round(float(high), 4),
                                  paired_delta_mean_ttft_at_2_qps_ms=mean_delta,
                                  different_trajectories=changed,
                                  paired_trajectories=count,
                                  resamples=RESAMPLES))
    out = confirm.OUT / 'analysis'
    save(out / 'points.csv', points)
    save(out / 'comparisons.csv', comparisons)
    save(out / 'paired_capacity_bootstrap_2s.csv', boots)
    lines = ['# SPARE 组件消融：24 窗口确认', '',
             'CPU fitted-clock simulator；每类负载 24 个预定配对窗口；'
             'KV=1024/2048/4096；2 s 容量单位 QPS。', '',
             '|Source|KV|'+'|'.join(confirm.ARMS)+'|',
             '|---|---:|'+'|'.join('---:' for _ in confirm.ARMS)+'|']
    for source in confirm.pilot.SOURCES:
        for kv in confirm.pilot.KVS:
            vals = [next(p['capacity_qps'] for p in points if
                    (p['source'], p['kv_blocks'], p['arm'], p['theta_s']) ==
                    (source, kv, arm, 2)) for arm in confirm.ARMS]
            lines.append(f"|{source}|{kv}|"+'|'.join(f'{v:.2f}' for v in vals)+'|')
    lines += ['', '完整 1/2/4/8 s 容量见 `points.csv`；2 s 容量配对区间、'
                  '共同 2 QPS 的 mean TTFT 差及轨迹差异见 '
                  '`paired_capacity_bootstrap_2s.csv`。', '']
    (out / 'REPORT_CN.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    analyze()
