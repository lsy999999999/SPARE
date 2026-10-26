"""Summarize paired component scans and exploratory capacity uncertainty."""
from __future__ import annotations

import csv

import numpy as np

from research.www_component_ablation_mean_20261019 import run

RESAMPLES = 2000
SEED = 20261019


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def records(source, kv, arm, rate):
    rows = run.old.prior.base.read(run.rate_path(source, kv, arm, rate))['rows']
    assert len(rows) == 12
    return {row['name']: row for row in rows}


def pair(source, kv, arm, rate):
    full = records(source, kv, 'spare', rate)
    other = records(source, kv, arm, rate)
    assert full.keys() == other.keys()
    valid = [name for name in full if full[name]['valid'] and other[name]['valid']]
    if len(valid) != 12:
        return None
    return sum(other[name]['mean_ttft_ms']-full[name]['mean_ttft_ms']
               for name in valid)/len(valid)


def capacity_bootstrap(source, kv, arm, names, indices):
    group = run.old.prior.base.read(run.OUT / 'groups' / f'{source}_c{kv}_{arm}.json')
    rates = sorted(group['rates'])
    passing = []
    for rate in rates:
        rows = records(source, kv, arm, rate)
        ordered = [rows[name] for name in names]
        valid = np.array([row['valid'] for row in ordered], dtype=bool)
        ttft = np.array([row.get('mean_ttft_ms', np.inf) for row in ordered], dtype=float)
        tbt = np.array([row.get('p99_tbt_ms', np.inf) for row in ordered], dtype=float)
        drift = np.array([row.get('backlog_drift', np.inf) for row in ordered], dtype=float)
        passed = (valid[indices].all(axis=1)
                  & (ttft[indices].mean(axis=1) <= 2000)
                  & (tbt[indices].mean(axis=1) <= 200)
                  & (drift[indices].mean(axis=1) <= max(2, rate)))
        passing.append(passed)
    continuous = np.logical_and.accumulate(np.stack(passing, axis=1), axis=1)
    return (continuous * np.asarray(rates)).max(axis=1)


def analyze():
    status = run.old.prior.base.read(run.OUT / 'status.json')
    assert status['completed'] == status['total'] and not status['errors']
    samples = run.prepare()
    rng = np.random.default_rng(SEED)
    points, comparisons, boots = [], [], []
    for source in run.SOURCES:
        names = sorted(s['name'] for s in samples if s['source'] == source)
        indices = rng.integers(0, len(names), size=(RESAMPLES, len(names)))
        for kv in run.KVS:
            groups = {arm: run.old.prior.base.read(run.OUT / 'groups' /
                      f'{source}_c{kv}_{arm}.json') for arm in run.ARMS}
            full_boot = capacity_bootstrap(source, kv, 'spare', names, indices)
            for arm in run.ARMS:
                for theta in run.THETA_MS:
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
                                                cap['last_pass_qps']-full_cap, 4),
                                            delta_mean_ttft_at_2_qps_ms=pair(source, kv, arm, 2.)))
                if arm == 'spare':
                    continue
                common = set(groups['spare']['rates']) & set(groups[arm]['rates'])
                changed = count = 0
                for rate in common:
                    full = records(source, kv, 'spare', rate)
                    other = records(source, kv, arm, rate)
                    assert full.keys() == other.keys()
                    changed += sum(full[name]['trajectory_sha256'] != other[name]['trajectory_sha256']
                                   for name in full if full[name]['valid'] and other[name]['valid'])
                    count += sum(full[name]['valid'] and other[name]['valid'] for name in full)
                variant_boot = capacity_bootstrap(source, kv, arm, names, indices)
                delta = variant_boot-full_boot
                lo, hi = np.quantile(delta, (.025, .975))
                boots.append(dict(source=source, kv_blocks=kv, arm=arm, theta_s=2,
                                  delta_capacity_qps=round(
                                      groups[arm]['capacities']['2000']['last_pass_qps']-
                                      groups['spare']['capacities']['2000']['last_pass_qps'], 4),
                                  bootstrap_low_qps=round(float(lo), 4),
                                  bootstrap_high_qps=round(float(hi), 4),
                                  variant_higher_fraction=round(float(np.mean(delta > 0)), 4),
                                  different_trajectories=changed,
                                  paired_trajectories=count,
                                  resamples=RESAMPLES))
    out = run.OUT / 'analysis'
    save_csv(out / 'points.csv', points)
    save_csv(out / 'comparisons.csv', comparisons)
    save_csv(out / 'paired_capacity_bootstrap_2s.csv', boots)
    display = ('spare', 'srp', 'constant1', 'constant8', 'constant16',
               'no_arrival', 'budget_only', 'chunk_family',
               'drop_candidate1', 'drop_candidate2', 'drop_candidate3',
               'drop_candidate4', 'no_finish_guard', 'no_decode_guard')
    lines = ['# SPARE 组件消融：12 窗口配对筛查', '',
             'CPU fitted-clock simulator；Code/Conversation 各 12 个预定窗口；'
             'KV=1024/2048/4096；容量门槛 mean TTFT=1/2/4/8 s，'
             'P99 TBT<=200 ms，积压<=max(2,QPS)。',
             '下表是 2 s 容量（QPS），并列出变体减 Full 的差值。', '',
             '|Source|KV|Variant|Capacity|Delta vs Full|',
             '|---|---:|---|---:|---:|']
    for source in run.SOURCES:
        for kv in run.KVS:
            full = next(p['capacity_qps'] for p in points if
                        (p['source'], p['kv_blocks'], p['arm'], p['theta_s']) ==
                        (source, kv, 'spare', 2))
            for arm in display:
                point = next(p for p in points if
                             (p['source'], p['kv_blocks'], p['arm'], p['theta_s']) ==
                             (source, kv, arm, 2))
                lines.append(f"|{source}|{kv}|{arm}|{point['capacity_qps']:.2f}|"
                             f"{point['capacity_qps']-full:+.2f}|")
    lines += ['', '完整 1/2/4/8 s 结果见 `points.csv`；共同 2 QPS 的配对 mean TTFT '
                  '见 `comparisons.csv`；12 窗口重采样区间和共同 QPS 轨迹差异见 '
                  '`paired_capacity_bootstrap_2s.csv`。',
              '这是一轮小样本组件筛查；接近零的差异需以更多配对窗口核实。', '']
    (out / 'REPORT_CN.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:10]))


if __name__ == '__main__':
    analyze()
