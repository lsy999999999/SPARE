"""Summarize the archived, matched 24-window arrival-term ablation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'artifacts/www_component_ablation_mean_20261020_r4'
OUT = ROOT / 'artifacts/www_arrival_term_review_20261020_r1'
RESAMPLES = 2000


def read(path):
    return json.loads(path.read_text())


def main():
    design = read(ARCHIVE / 'design.json')
    status = read(ARCHIVE / 'status.json')
    assert status['completed'] == status['total'] and not status['errors']
    assert len(design['samples']) == 48
    assert design['capacity_rule'].startswith('all 24 valid')
    policy = ROOT / 'research/www_component_ablation_mean_20261019/policy.py'
    assert hashlib.sha256(policy.read_bytes()).hexdigest() == design['policy_sha256']
    rng = np.random.default_rng(20261020)
    capacity_intervals = {
        (r['source'], int(r['kv_blocks'])): r
        for r in csv.DictReader((ARCHIVE / 'analysis/paired_capacity_bootstrap_2s.csv').open())
        if r['arm'] == 'no_arrival'
    }
    output = []
    for source in ('code', 'conversation'):
        names = sorted(s['name'] for s in design['samples'] if s['source'] == source)
        assert len(names) == 24
        draws = rng.integers(0, 24, size=(RESAMPLES, 24))
        for kv in (1024, 2048, 4096):
            records, groups = {}, {}
            for arm in ('spare', 'no_arrival'):
                path = ARCHIVE / 'rates' / f'{source}_c{kv}_{arm}_r2.json'
                records[arm] = {r['name']: r for r in read(path)['rows']}
                assert sorted(records[arm]) == names
                assert all(r['valid'] and r['capacity'] == kv and r['rate'] == 2
                           for r in records[arm].values())
                groups[arm] = read(ARCHIVE / 'groups' / f'{source}_c{kv}_{arm}.json')
            full, removed = records['spare'], records['no_arrival']
            ttft = np.array([removed[n]['mean_ttft_ms']-full[n]['mean_ttft_ms']
                             for n in names])
            low, high = np.quantile(ttft[draws].mean(axis=1), (.025, .975))
            cap = capacity_intervals[source, kv]
            row = dict(source=source, kv_blocks=kv,
                       full_ttft_2qps_ms=np.mean([full[n]['mean_ttft_ms'] for n in names]),
                       no_arrival_ttft_2qps_ms=np.mean([removed[n]['mean_ttft_ms'] for n in names]),
                       ttft_delta_ms=ttft.mean(), ttft_delta_low_ms=low,
                       ttft_delta_high_ms=high,
                       changed_trajectories_2qps=sum(
                           full[n]['trajectory_sha256'] != removed[n]['trajectory_sha256']
                           for n in names),
                       full_preemptions_per_window=np.mean([full[n]['preemptions'] for n in names]),
                       no_arrival_preemptions_per_window=np.mean(
                           [removed[n]['preemptions'] for n in names]),
                       full_recomputed_tokens_per_window=np.mean(
                           [full[n]['recomputed_tokens'] for n in names]),
                       no_arrival_recomputed_tokens_per_window=np.mean(
                           [removed[n]['recomputed_tokens'] for n in names]),
                       full_capacity_2s_qps=groups['spare']['capacities']['2000']['last_pass_qps'],
                       no_arrival_capacity_2s_qps=groups['no_arrival']['capacities']['2000']['last_pass_qps'],
                       capacity_delta_2s_qps=float(cap['delta_capacity_qps']),
                       capacity_delta_low_qps=float(cap['bootstrap_low_qps']),
                       capacity_delta_high_qps=float(cap['bootstrap_high_qps']),
                       full_capacity_4s_qps=groups['spare']['capacities']['4000']['last_pass_qps'],
                       no_arrival_capacity_4s_qps=groups['no_arrival']['capacities']['4000']['last_pass_qps'])
            assert round(row['no_arrival_capacity_2s_qps'] - row['full_capacity_2s_qps'], 4) == row['capacity_delta_2s_qps']
            output.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'paired_24windows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    lines = ['# 到达项消融：24 窗口配对核对', '',
             '`no_arrival` 将门槛公式中的 `tau` 设为 0；候选、保护和省块接受判断均保持。'
             '该项删除的是未来到达负载估计的影响，门槛仍根据当前状态变化。'
             '在相同状态下，`tau=0` 使门槛不低于 Full，因此候选更难达到省块要求。', '',
             'Code/Conversation 各 24 个相同窗口，KV=1024/2048/4096；CPU 拟合时钟模拟。'
             '延迟在共同 2 QPS 下测量；容量要求 mean TTFT 不超过 2 s，'
             '并满足原 P99 TBT 和积压约束。差值均为去到达项减 Full。', '',
             '|负载|KV|Full/去到达 mean TTFT (ms)|差值及 95% 配对区间 (ms)|'
             '不同轨迹/24|Full/去到达 2 s 容量 (QPS)|容量差及 95% 区间 (QPS)|'
             'Full/去到达 4 s 容量 (QPS)|',
             '|---|---:|---:|---|---:|---:|---|---:|']
    for r in output:
        lines.append(f"|{r['source']}|{r['kv_blocks']}|"
                     f"{r['full_ttft_2qps_ms']:.2f}/{r['no_arrival_ttft_2qps_ms']:.2f}|"
                     f"{r['ttft_delta_ms']:+.2f} "
                     f"[{r['ttft_delta_low_ms']:+.2f}, {r['ttft_delta_high_ms']:+.2f}]|"
                     f"{r['changed_trajectories_2qps']}|"
                     f"{r['full_capacity_2s_qps']:.2f}/{r['no_arrival_capacity_2s_qps']:.2f}|"
                     f"{r['capacity_delta_2s_qps']:+.2f} "
                     f"[{r['capacity_delta_low_qps']:+.2f}, {r['capacity_delta_high_qps']:+.2f}]|"
                     f"{r['full_capacity_4s_qps']:.2f}/{r['no_arrival_capacity_4s_qps']:.2f}|")
    lines += ['', 'KV=1024 时，去到达项的 TTFT 增加，配对区间均不跨零。'
              'KV=2048 的 TTFT 变化较小，KV=4096、2 QPS 的配对轨迹完全相同。'
              '六组 2 s 容量差的区间均跨零；4 s 容量的方向并不一致。'
              '因此该数据支持到达项在紧张 KV 下的延迟作用，'
              '尚不支持全局容量收益。', '',
              '2 QPS 的 KV=1024 每窗口平均抢占次数：Code '
              f"{output[0]['full_preemptions_per_window']:.2f}/"
              f"{output[0]['no_arrival_preemptions_per_window']:.2f}，Conversation "
              f"{output[3]['full_preemptions_per_window']:.2f}/"
              f"{output[3]['no_arrival_preemptions_per_window']:.2f}（Full/去到达）。"
              '这只是闭环关联，不等同于同一状态下的因果归因。', '',
              '新模拟次数：0。TTFT 区间按 24 个配对窗口重采样 2,000 次；'
              '容量区间直接引用既有 24 窗口归档，不能将 4 s 点估计解释为已验证显著差异。'
              '原始逐窗口数据：`../www_component_ablation_mean_20261020_r4/rates/`；'
              '容量扫描：`../www_component_ablation_mean_20261020_r4/groups/`；'
              '完整数值与重算量见 [paired_24windows.csv](paired_24windows.csv)。', '']
    (OUT / 'REPORT_CN.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
