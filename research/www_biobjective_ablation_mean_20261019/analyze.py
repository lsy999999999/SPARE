"""Summarize matched mean-TTFT and SLO-capacity component comparisons."""
from __future__ import annotations

import csv
from pathlib import Path

from research.www_biobjective_ablation_mean_20261019 import run


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_delta(source, kv, arm, qps):
    full = run.prior.base.read(run.rate_path(source, kv, 'spare', qps))
    other = run.prior.base.read(run.rate_path(source, kv, arm, qps))
    a = {r['name']: r for r in full['rows']}
    b = {r['name']: r for r in other['rows']}
    assert a.keys() == b.keys()
    values = [b[key]['mean_ttft_ms'] - a[key]['mean_ttft_ms']
              for key in a if a[key]['valid'] and b[key]['valid']]
    return sum(values) / len(values) if len(values) == 50 else None


def analyze():
    design = run.prepare()
    status = run.prior.base.read(run.OUT / 'status.json')
    assert status['completed'] == status['total'] == len(run.KVS) * len(run.SOURCES) * len(run.ARMS)
    assert not status['errors']
    points, comparisons = [], []
    for source in run.SOURCES:
        for kv in run.KVS:
            groups = {arm: run.prior.base.read(run.OUT / 'groups' /
                      f'{source}_c{kv}_{arm}.json') for arm in run.ARMS}
            for arm in run.ARMS:
                ref = run.prior.base.read(run.rate_path(source, kv, arm, .5))['summary']
                for theta in run.THETA_MS:
                    cap = groups[arm]['capacities'][f'{theta:g}']
                    points.append(dict(source=source, kv_blocks=kv, arm=arm,
                                       theta_s=theta / 1000,
                                       mean_ttft_at_0_5_qps_ms=ref['mean_ttft_ms'],
                                       capacity_qps=cap['last_pass_qps'],
                                       next_fail_qps=cap['first_fail_qps'],
                                       resolution_qps=cap['resolution_qps'],
                                       failure_causes=';'.join(cap['failure_causes'])))
                    if arm == 'spare':
                        continue
                    full = groups['spare']['capacities'][f'{theta:g}']
                    comparisons.append(dict(source=source, kv_blocks=kv, arm=arm,
                                            theta_s=theta / 1000,
                                            capacity_delta_vs_spare_qps=cap['last_pass_qps']-
                                            full['last_pass_qps'],
                                            capacity_ratio_vs_spare=(cap['last_pass_qps'] /
                                                full['last_pass_qps'] if full['last_pass_qps'] else None),
                                            paired_delta_ttft_0_5_ms=pair_delta(source, kv, arm, .5),
                                            paired_delta_ttft_2_ms=pair_delta(source, kv, arm, 2.)
                                            if 2. in groups[arm]['rates'] and 2. in groups['spare']['rates']
                                            else None))
    out = run.OUT / 'analysis'
    save_csv(out / 'points.csv', points)
    save_csv(out / 'comparisons.csv', comparisons)
    lines = ['# 新版 mean TTFT 容量消融', '',
             f'CPU 拟合时钟模拟；Code/Conversation 各 50 个配对测试窗口；KV={list(run.KVS)}。',
             '容量要求 50 窗全部有效、mean TTFT 不超过指定门槛、窗口均值 P99 TBT <= 200 ms、',
             '积压增长 <= max(2, QPS)。容量是离散 QPS 网格的连续通过点估计。',
             '下表的 TTFT 差为变体减 Full SPARE，正值表示变体更慢。', '',
             '|Workload|KV|Variant|Capacity at 2 s (QPS)|Delta vs Full (QPS)|',
             '|---|---:|---|---:|---:|']
    for row in comparisons:
        if row['theta_s'] != 2:
            continue
        arm = row['arm']
        point = next(p for p in points if (p['source'],p['kv_blocks'],p['arm'],p['theta_s']) ==
                     (row['source'],row['kv_blocks'],arm,2))
        lines.append(f"|{row['source']}|{row['kv_blocks']}|{arm}|"
                     f"{point['capacity_qps']:.2f}|{row['capacity_delta_vs_spare_qps']:+.2f}|")
    lines += ['', '完整数值、所有 TTFT 门槛与失败原因见 `points.csv`；'
                  '共同 QPS 的配对 TTFT 差见 `comparisons.csv`。',
              '关闭暂缓后再关闭保护与 SRP 等价；该重复格不作为独立消融证据。',
              '各保护/候选变体属于 CPU 模拟结果，尚不能直接写作 GPU 容量收益。', '']
    (out / 'REPORT_CN.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:11]))


if __name__ == '__main__':
    analyze()
