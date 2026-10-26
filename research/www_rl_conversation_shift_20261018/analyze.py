"""Audit and summarize the Conversation-only RL transfer experiment."""
from __future__ import annotations

import csv
import json
from statistics import mean

from research.www_rl_conversation_shift_20261018 import run as experiment
from research.www_rl_conversation_shift_20261018.followup import KVS
from research.www_rl_mixed_generalization_20261017 import inputs, sweep


OUT = experiment.OUT
OLD = inputs.OUT
ANALYSIS = OUT / 'analysis'
SEEDS = experiment.SEEDS


def read(path):
    with path.open() as stream:
        return json.load(stream)


def write_csv(name, rows):
    with (ANALYSIS / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frontier_group(source, arm, seed):
    path = OUT / 'frontier' / 'groups' / (
        f'original_{source}_c2048_{arm}_seed{seed}.json')
    group = read(path)
    assert group['source'] == source and group['arm'] == arm
    assert group['seed'] == seed and group['kv_blocks'] == 2048
    assert len(group['thresholds']) == len(experiment.frontier.LIMITS_MS)
    for row in group['thresholds']:
        if not row['capacity_qps']:
            continue
        assert row['attained_mean_ttft_ms'] <= row['mean_ttft_limit_ms']
        assert row['attained_p99_tbt_ms'] <= 200
        assert row['attained_backlog_drift'] <= max(2, row['capacity_qps'])
    return group


def frontier_rows():
    output = []
    for source in ('conversation', 'code'):
        spare = frontier_group(source, 'spare', 0)
        rl = [frontier_group(source, 'rl_mixed', seed) for seed in SEEDS]
        assert all(group['input_sha256'] == spare['input_sha256'] for group in rl)
        for index, limit in enumerate(experiment.frontier.LIMITS_MS):
            a = spare['thresholds'][index]
            b = [group['thresholds'][index] for group in rl]
            assert all(row['mean_ttft_limit_ms'] == limit for row in [a, *b])
            assert all(row['resolution_qps'] is None or row['resolution_qps'] <= .020001
                       for row in [a, *b])
            capacities = [row['capacity_qps'] for row in b]
            spare_qps = a['capacity_qps']
            output.append(dict(source=source, kv_blocks=2048,
                               mean_ttft_limit_ms=limit,
                               spare_qps=spare_qps,
                               rl_mean_qps=mean(capacities),
                               rl_seed1017_qps=capacities[0],
                               rl_seed2027_qps=capacities[1],
                               rl_seed3037_qps=capacities[2],
                               rl_gain_pct=(100 * (mean(capacities) / spare_qps - 1)
                                            if spare_qps else None),
                               windows=50, input_sha256=spare['input_sha256']))
    return output


def group_for_kv(kv, arm, seed):
    root = OLD if arm == 'spare' else OUT
    path = root / 'groups' / f'original_conversation_c{kv}_{arm}_seed{seed}.json'
    group = read(path)
    assert group['stage'] == 'complete' and group['kv_blocks'] == kv
    assert group['source'] == 'conversation' and group['seed'] == seed
    assert group['resolution_qps'] <= .020001
    return group


def kv_rows():
    output = []
    for kv in (2048, *KVS):
        if kv == 2048:
            steady = [row for row in frontier_rows()
                      if row['source'] == 'conversation'
                      and row['mean_ttft_limit_ms'] == 2000][0]
            output.append(dict(kv_blocks=kv, spare_qps=steady['spare_qps'],
                               rl_mean_qps=steady['rl_mean_qps'],
                               rl_seed1017_qps=steady['rl_seed1017_qps'],
                               rl_seed2027_qps=steady['rl_seed2027_qps'],
                               rl_seed3037_qps=steady['rl_seed3037_qps'],
                               rl_gain_pct=steady['rl_gain_pct'],
                               input_sha256=steady['input_sha256']))
            continue
        spare = group_for_kv(kv, 'spare', 0)
        rl = [group_for_kv(kv, 'rl_mixed', seed) for seed in SEEDS]
        assert all(group['input_sha256'] == spare['input_sha256'] for group in rl)
        values = [group['capacity_qps'] for group in rl]
        output.append(dict(kv_blocks=kv, spare_qps=spare['capacity_qps'],
                           rl_mean_qps=mean(values),
                           rl_seed1017_qps=values[0], rl_seed2027_qps=values[1],
                           rl_seed3037_qps=values[2],
                           rl_gain_pct=100 * (mean(values) / spare['capacity_qps'] - 1),
                           input_sha256=spare['input_sha256']))
    return output


def burst_group(duty, arm, seed):
    root = OLD if arm == 'spare' else OUT
    path = root / 'burst_arrival_20261018' / 'groups' / (
        f'conversation_d{duty:g}_c2048_{arm}_seed{seed}.json')
    group = read(path)
    assert group['stage'] == 'complete' and group['duty'] == duty
    assert group['source'] == 'conversation' and group['kv_blocks'] == 2048
    assert group['capacity_qps'] == 0 or group['resolution_qps'] <= .020001
    failure = group['first_failure_summary']
    assert failure['failed_samples'] == 0
    assert failure['p99_tbt_ms'] <= 200
    assert failure['backlog_drift'] <= max(2, group['next_failed_qps'])
    assert failure['mean_ttft_ms'] > 2000
    return group


def burst_rows(steady):
    output = []
    for duty in (.5, .2):
        spare = burst_group(duty, 'spare', 0)
        rl = [burst_group(duty, 'rl_mixed', seed) for seed in SEEDS]
        assert all(group['input_sha256'] == spare['input_sha256'] for group in rl)
        values = [group['capacity_qps'] for group in rl]
        output.append(dict(duty_fraction=duty, instantaneous_multiplier=1 / duty,
                           spare_qps=spare['capacity_qps'], rl_mean_qps=mean(values),
                           rl_seed1017_qps=values[0], rl_seed2027_qps=values[1],
                           rl_seed3037_qps=values[2],
                           rl_gain_pct=100 * (mean(values) / spare['capacity_qps'] - 1),
                           spare_drop_pct=100 * (1 - spare['capacity_qps'] /
                                                   steady['spare_qps']),
                           rl_drop_pct=100 * (1 - mean(values) / steady['rl_mean_qps']),
                           input_sha256=spare['input_sha256']))
    return output


def test_distribution(source):
    samples = [s for s in experiment.mixed.old.samples('test')
               if s['source'] == source]
    requests = [row for sample in samples
                for row in read(experiment.ROOT / sample['input'])]
    return mean(row['prompt'] for row in requests), mean(
        row['output'] for row in requests)


def report(frontier, kv, burst):
    at = lambda source, limit: next(row for row in frontier
                                   if row['source'] == source
                                   and row['mean_ttft_limit_ms'] == limit)
    conv = at('conversation', 2000)
    code = at('code', 2000)
    cp, co = test_distribution('conversation')
    kp, ko = test_distribution('code')
    lines = ['# Conversation-only RL 跨负载与扰动实验', '',
             'CPU simulator；Double DQN 在 KV=2048 的 16 个 Conversation 窗口上'
             '训练 1024 回合，另 4 个 Conversation 窗口选检查点。Code 完全留出；'
             '三个种子均选择第 1024 回合。测试时冻结策略，每条件 50 个窗口。', '',
             '容量要求 mean TTFT 不超过对应门槛、P99 TBT ≤ 200 ms、'
             '积压增长 ≤ max(2, 输入 QPS)，且所有窗口有效；容量边界细化到 0.02 QPS。', '',
             '## 跨工作负载迁移', '',
             f'测试池平均 prompt/output 长度：Conversation {cp:.0f}/{co:.0f} tokens；'
             f'Code {kp:.0f}/{ko:.0f} tokens。两者同时改变输入与输出长度。', '',
             '| mean TTFT 门槛 | Conversation：SPARE / RL / RL 优势 | '
             'Code：SPARE / RL / RL 优势 |',
             '|---:|---:|---:|']
    for limit in (500, 1000, 1500, 2000, 3000, 4000):
        a, b = at('conversation', limit), at('code', limit)
        lines.append(f'|{limit} ms|{a["spare_qps"]:.2f} / '
                     f'{a["rl_mean_qps"]:.3f} / {a["rl_gain_pct"]:+.2f}%|'
                     f'{b["spare_qps"]:.2f} / {b["rl_mean_qps"]:.3f} / '
                     f'{b["rl_gain_pct"]:+.2f}%|')
    lines.extend(['', f'在 2 s 门槛下，RL 在训练来源 Conversation 上比 SPARE 高 '
                  f'{conv["rl_gain_pct"]:+.2f}%，转到留出的 Code 后为 '
                  f'{code["rl_gain_pct"]:+.2f}%；相对收益变化 '
                  f'{code["rl_gain_pct"]-conv["rl_gain_pct"]:+.2f} 个百分点。'
                  '500 ms 门槛下 Code 仍领先，因此不能写成所有 TTFT 门槛均退化。'
                  'Code/Conversation 的 prompt 和 output 分布都不同，'
                  '该实验检验整体工作负载迁移，不能单独归因于某一种长度变化。', '',
                  '[全部 15 个门槛及逐种子容量](frontier_capacity.csv)。'])
    if kv:
        lines.extend(['', '## 跨 KV 配置', '',
                      '训练 KV 固定 2048 blocks；测试仍用 Conversation，'
                      'DQN 状态含总 KV 与剩余 KV。容量门槛为 mean TTFT ≤ 2 s。', '',
                      '| KV blocks | SPARE QPS | RL 三种子均值 QPS | RL 相对 SPARE |',
                      '|---:|---:|---:|---:|'])
        for row in kv:
            lines.append(f'|{row["kv_blocks"]}|{row["spare_qps"]:.2f}|'
                         f'{row["rl_mean_qps"]:.3f}|{row["rl_gain_pct"]:+.2f}%|')
        lines.extend(['', '[逐种子容量](cross_kv_capacity.csv)。'])
    else:
        lines.extend(['', '## 跨 KV 配置', '', '测试尚未完成。'])
    if burst:
        lines.extend(['', '## 突发到达', '',
                      '固定 KV=2048、Conversation 请求长度与每 10 s 周期请求数；'
                      '把周期内到达压缩到前 5 s 或 2 s，形成 2×/5× 局部强度。'
                      '容量门槛为 mean TTFT ≤ 2 s。', '',
                      '| 局部强度 | SPARE QPS | RL 三种子均值 QPS | '
                      'RL 相对 SPARE | SPARE 下降 | RL 下降 |',
                      '|---:|---:|---:|---:|---:|---:|'])
        for row in burst:
            lines.append(f'|{row["instantaneous_multiplier"]:g}×|'
                         f'{row["spare_qps"]:.2f}|{row["rl_mean_qps"]:.3f}|'
                         f'{row["rl_gain_pct"]:+.2f}%|'
                         f'{row["spare_drop_pct"]:.2f}%|'
                         f'{row["rl_drop_pct"]:.2f}%|')
        lines.extend(['', '[逐种子容量](burst_capacity.csv)。'])
    else:
        lines.extend(['', '## 突发到达', '', '测试尚未完成。'])
    lines.extend(['', '三个种子的范围反映训练随机性，不是测试窗口置信区间。'
                  '这些结果是 CPU simulator 的容量点估计。RL 额外需要训练与选点；'
                  '已有同机测量没有显示明确的在线调度时间差异。'])
    (ANALYSIS / 'REPORT_CN.md').write_text('\n'.join(lines) + '\n')


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    frontier = frontier_rows()
    assert len(frontier) == 30
    write_csv('frontier_capacity.csv', frontier)
    ready_kv = all((OUT / 'groups' /
                    f'original_conversation_c{kv}_rl_mixed_seed{seed}.json').exists()
                   for kv in KVS for seed in SEEDS)
    ready_burst = all((OUT / 'burst_arrival_20261018' / 'groups' /
                       f'conversation_d{duty:g}_c2048_rl_mixed_seed{seed}.json').exists()
                      for duty in (.5, .2) for seed in SEEDS)
    kv = kv_rows() if ready_kv else None
    burst = burst_rows(next(row for row in frontier if row['source'] == 'conversation'
                            and row['mean_ttft_limit_ms'] == 2000)) if ready_burst else None
    if kv:
        write_csv('cross_kv_capacity.csv', kv)
    if burst:
        write_csv('burst_capacity.csv', burst)
    report(frontier, kv, burst)
    print(f'frontier={len(frontier)} rows, kv={len(kv) if kv else "running"}, '
          f'burst={len(burst) if burst else "running"}')


if __name__ == '__main__':
    main()
