"""Audit paired step-down traces and report transient response."""
from __future__ import annotations

import csv
from statistics import mean

import numpy as np

from research.www_rl_conversation_shift_20261018 import step_down


OUT = step_down.OUT
SEEDS = step_down.SEEDS


def group(ratio, arm, seed):
    path = OUT / 'groups' / f'conversation_ratio{ratio:g}_{arm}_seed{seed}.json'
    data = step_down.mixed.old.base.read(path)
    assert data['stage'] == 'complete'
    assert (data['ratio'], data['arm'], data['seed']) == (ratio, arm, seed)
    assert data['pre_qps'] == step_down.PRE_QPS
    assert data['post_qps'] == step_down.PRE_QPS * ratio
    assert len(data['rows']) == 50
    return data


def main():
    status = step_down.mixed.old.base.read(OUT / 'status.json')
    assert status['complete'] and status['completed'] == 12
    records = {}
    for ratio in step_down.RATIOS:
        records[ratio, 'spare', 0] = group(ratio, 'spare', 0)
        for seed in SEEDS:
            records[ratio, 'rl_mixed', seed] = group(ratio, 'rl_mixed', seed)
        reference = records[ratio, 'spare', 0]
        for seed in SEEDS:
            candidate = records[ratio, 'rl_mixed', seed]
            assert candidate['input_sha256'] == reference['input_sha256']
            assert candidate['stream_sha256'] == reference['stream_sha256']
            assert [row['name'] for row in candidate['rows']] == [
                row['name'] for row in reference['rows']]
            assert [row['arrivals'] for row in candidate['rows']] == [
                row['arrivals'] for row in reference['rows']]
        assert all(item['summary']['failed_windows'] == 0 for item in
                   [reference, *(records[ratio, 'rl_mixed', s] for s in SEEDS)])
    for arm, seed in [('spare', 0), *(('rl_mixed', s) for s in SEEDS)]:
        baseline = records[1., arm, seed]
        for ratio in (.5, .2):
            reduced = records[ratio, arm, seed]
            assert [row['outstanding_at_30s'] for row in baseline['rows']] == [
                row['outstanding_at_30s'] for row in reduced['rows']]

    rows = []
    rng = np.random.default_rng(20261018)
    resamples = rng.integers(0, 50, size=(10000, 50))

    def window_values(ratio, arm, seed, phase):
        return np.array([row[phase]['mean_ttft_ms']
                         for row in records[ratio, arm, seed]['rows']])

    def paired_interval(spare_values, rl_values):
        changes = 100 * (rl_values[resamples].mean(axis=1) /
                         spare_values[resamples].mean(axis=1) - 1)
        return tuple(np.percentile(changes, (2.5, 97.5)))

    for ratio in step_down.RATIOS:
        spare = records[ratio, 'spare', 0]['summary']
        rl = [records[ratio, 'rl_mixed', seed]['summary'] for seed in SEEDS]
        output = dict(pre_qps=step_down.PRE_QPS,
                      post_qps=step_down.PRE_QPS * ratio,
                      ratio=ratio, windows=50,
                      mean_post_arrivals=spare['post_mean_requests'],
                      spare_post_mean_ttft_ms=spare['post_mean_ttft_ms'],
                      rl_post_mean_ttft_ms=mean(x['post_mean_ttft_ms'] for x in rl),
                      spare_late_mean_ttft_ms=spare['late_mean_ttft_ms'],
                      rl_late_mean_ttft_ms=mean(x['late_mean_ttft_ms'] for x in rl),
                      spare_late_p95_ttft_ms=spare['late_p95_ttft_ms'],
                      rl_late_p95_ttft_ms=mean(x['late_p95_ttft_ms'] for x in rl),
                      spare_outstanding_30=spare['mean_outstanding_at_30s'],
                      rl_outstanding_30=mean(x['mean_outstanding_at_30s'] for x in rl),
                      spare_outstanding_45=spare['mean_outstanding_at_45s'],
                      rl_outstanding_45=mean(x['mean_outstanding_at_45s'] for x in rl),
                      spare_outstanding_60=spare['mean_outstanding_at_60s'],
                      rl_outstanding_60=mean(x['mean_outstanding_at_60s'] for x in rl),
                      spare_p99_tbt_ms=spare['mean_p99_tbt_ms'],
                      rl_p99_tbt_ms=mean(x['mean_p99_tbt_ms'] for x in rl))
        output['rl_post_ttft_change_pct'] = 100 * (
            output['rl_post_mean_ttft_ms'] / output['spare_post_mean_ttft_ms'] - 1)
        output['rl_late_ttft_change_pct'] = 100 * (
            output['rl_late_mean_ttft_ms'] / output['spare_late_mean_ttft_ms'] - 1)
        for phase in ('post', 'late'):
            spare_values = window_values(ratio, 'spare', 0, phase)
            rl_values = np.mean([window_values(ratio, 'rl_mixed', seed, phase)
                                 for seed in SEEDS], axis=0)
            low, high = paired_interval(spare_values, rl_values)
            output[f'rl_{phase}_ttft_change_ci_low_pct'] = low
            output[f'rl_{phase}_ttft_change_ci_high_pct'] = high
        if ratio != 1.:
            spare_current = window_values(ratio, 'spare', 0, 'post')
            spare_steady = window_values(1., 'spare', 0, 'post')
            rl_current = np.mean([window_values(ratio, 'rl_mixed', seed, 'post')
                                  for seed in SEEDS], axis=0)
            rl_steady = np.mean([window_values(1., 'rl_mixed', seed, 'post')
                                 for seed in SEEDS], axis=0)
            gap = 100 * (rl_current.mean() / rl_steady.mean() -
                         spare_current.mean() / spare_steady.mean())
            sampled = 100 * (
                rl_current[resamples].mean(axis=1) /
                rl_steady[resamples].mean(axis=1) -
                spare_current[resamples].mean(axis=1) /
                spare_steady[resamples].mean(axis=1))
            output['relative_recovery_gap_pp'] = gap
            output['relative_recovery_ci_low_pp'], \
                output['relative_recovery_ci_high_pp'] = np.percentile(
                    sampled, (2.5, 97.5))
        else:
            output['relative_recovery_gap_pp'] = None
            output['relative_recovery_ci_low_pp'] = None
            output['relative_recovery_ci_high_pp'] = None
        output.update({f'rl_seed{seed}_post_mean_ttft_ms':
                       records[ratio, 'rl_mixed', seed]['summary']['post_mean_ttft_ms']
                       for seed in SEEDS})
        rows.append(output)

    analysis = OUT / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    with (analysis / 'step_down.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Conversation 负载阶跃下降：SPARE 与冻结 RL', '',
             'KV=2048；先以 4.5 QPS 到达，在第 30 秒保持原速或降到 '
             '2.25/0.9 QPS。保持同一批 50 个测试窗口、相同请求长度和配对到达流；'
             'RL 为三个种子均值。这里测响应过程，不定义非稳态 serving capacity。', '',
             '切换后的 mean TTFT 统计 30–60 s 到达请求；后段 mean TTFT 统计 '
             '45–60 s 到达请求；未完成数分别在 30/45/60 s 采样。', '',
             '| 切换后 QPS | SPARE / RL 切换后 mean TTFT | '
             'SPARE / RL 后段 mean TTFT | SPARE / RL 在 60 s 的未完成数 |',
             '|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'|{row["post_qps"]:.2f}|'
                     f'{row["spare_post_mean_ttft_ms"]:.0f} / '
                     f'{row["rl_post_mean_ttft_ms"]:.0f} ms|'
                     f'{row["spare_late_mean_ttft_ms"]:.0f} / '
                     f'{row["rl_late_mean_ttft_ms"]:.0f} ms|'
                     f'{row["spare_outstanding_60"]:.1f} / '
                     f'{row["rl_outstanding_60"]:.1f}|')
    lines.extend(['', '切换前到达流相同；逐窗口 30 s 未完成数已验证在每种方法的'
                  '三个比率之间完全一致。每种比率下 SPARE 与 RL 的请求流哈希一致，'
                  '12 个组均为 50/50 有效窗口。'
                  '这里的“降到 1/2、1/5”是 30 s 后持续的到达率阶跃，'
                  '不同于先前每 10 s 内局部压缩请求的周期性 2×/5× 突发。', '',
                  '配对窗口 bootstrap 95% 区间（固定三个训练种子）：', '',
                  '| 切换后 QPS | RL 相对 SPARE 的切换后 mean TTFT 变化 | '
                  'RL 与 SPARE 的相对降载收益差 |',
                  '|---:|---:|---:|'])
    for row in rows:
        gap = (f'{row["relative_recovery_gap_pp"]:+.2f} pp '
               f'[{row["relative_recovery_ci_low_pp"]:+.2f}, '
               f'{row["relative_recovery_ci_high_pp"]:+.2f}]'
               if row['relative_recovery_gap_pp'] is not None else '对照组')
        lines.append(f'|{row["post_qps"]:.2f}|'
                     f'{row["rl_post_ttft_change_pct"]:+.2f}% '
                     f'[{row["rl_post_ttft_change_ci_low_pct"]:+.2f}, '
                     f'{row["rl_post_ttft_change_ci_high_pct"]:+.2f}]|{gap}|')
    lines.extend(['', '相对降载收益差比较各方法降载后的 mean TTFT / 不降载对照值；'
                  '负值表示 RL 的比例下降更多。该区间反映测试窗口变化，'
                  '不包含重新训练的随机性。', '',
                  '[逐种子 TTFT、P95 TTFT 与未完成请求数](step_down.csv)。'])
    (analysis / 'REPORT_CN.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
