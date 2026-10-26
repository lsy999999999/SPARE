"""Audit and tabulate paired burst-capacity results without plotting."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from research.www_rl_mixed_generalization_20261017 import burst, run, sweep


SEEDS = (1017, 2027, 3037)
SOURCES = ('code', 'conversation')
DUTIES = (.5, .2)
ANALYSIS = burst.OUT / 'analysis'


def read(path):
    with path.open() as stream:
        return json.load(stream)


def group_path(source, duty, arm, seed):
    return (burst.OUT / 'groups' /
            f'{source}_d{duty:g}_c{burst.KV}_{arm}_seed{seed}.json')


def steady_capacity(source, arm, seed):
    path = (run.OUT / 'groups' /
            f'original_{source}_c{burst.KV}_{arm}_seed{seed}.json')
    result = read(path)
    assert result['stage'] == 'complete'
    return result['capacity_qps']


def verify_group(source, duty, arm, seed):
    result = read(group_path(source, duty, arm, seed))
    assert result['stage'] == 'complete'
    assert (result['source'], result['duty'], result['arm'], result['seed']) == (
        source, duty, arm, seed)
    assert result['input_sha256'] == sweep.source_hash(
        sweep.cohort_for('original', source))
    passed, failed = result['capacity_qps'], result['next_failed_qps']
    assert passed == 0 or failed - passed <= .020001
    assert result['below_minimum_measured_rate'] == (passed == 0)
    failure = result['first_failure_summary']
    assert failure['failed_samples'] == 0
    assert failure['p99_tbt_ms'] <= 200
    assert failure['backlog_drift'] <= max(2, failed)
    assert failure['mean_ttft_ms'] > sweep.LIMIT_MS
    for rate, expected in ((passed, True), (failed, False)):
        if rate == 0:
            continue
        record = read(burst.rate_path(source, duty, arm, seed, rate))
        assert record['input_sha256'] == result['input_sha256']
        assert record['duty'] == duty and len(record['rows']) == 50
        assert record['summary'] == run.old.prior.old.summary(record['rows'], rate)
        assert sweep.passed(record['summary'], rate) == expected
    return result


def rows():
    output = []
    for source in SOURCES:
        spare_steady = steady_capacity(source, 'spare', 0)
        rl_steady = [steady_capacity(source, 'rl_mixed', seed) for seed in SEEDS]
        for duty in DUTIES:
            spare = verify_group(source, duty, 'spare', 0)['capacity_qps']
            rl = [verify_group(source, duty, 'rl_mixed', seed)['capacity_qps']
                  for seed in SEEDS]
            output.append(dict(source=source, kv_blocks=burst.KV,
                               burst_period_s=burst.PERIOD_MS / 1000,
                               duty_fraction=duty,
                               instantaneous_multiplier=1 / duty,
                               spare_steady_qps=spare_steady,
                               rl_steady_mean_qps=mean(rl_steady),
                               spare_burst_qps=spare,
                               rl_burst_mean_qps=mean(rl),
                               rl_burst_min_qps=min(rl),
                               rl_burst_max_qps=max(rl),
                               rl_seed1017_qps=rl[0], rl_seed2027_qps=rl[1],
                               rl_seed3037_qps=rl[2],
                               rl_vs_spare_pct=100 * (mean(rl) / spare - 1)
                               if spare else None,
                               spare_drop_pct=100 * (1 - spare / spare_steady),
                               rl_drop_pct=100 * (1 - mean(rl) / mean(rl_steady)),
                               windows_per_source=50))
    return output


def main():
    expected = 2 * 2 * 4
    assert len(list((burst.OUT / 'groups').glob('*.json'))) == expected
    data = rows()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    with (ANALYSIS / 'capacity_burst.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=data[0])
        writer.writeheader()
        writer.writerows(data)
    lines = [
        '# RL-Mixed 突发到达容量补充实验', '',
        'KV=2048；mean TTFT <= 2 s、P99 TBT <= 200 ms、积压增长 <= max(2, 输入 QPS)；'
        '每个条件使用同一批 50 个测试窗口，容量边界细化到 0.02 QPS。', '',
        '每个原始 Poisson 到达流按 10 s 周期变换：保持请求 ID、prompt/output 长度、'
        '请求数与周期归属，只把周期内到达压缩到前 5 s（2 倍局部强度）或前 2 s'
        '（5 倍局部强度）。SPARE 与冻结的三个 RL-Mixed 种子使用完全相同的变换后请求。'
        '稳态容量来自此前相同 50 窗口的 2 s 扫描。', '',
        '| 负载 | 局部强度 | SPARE 稳态 / 突发 QPS | RL 稳态 / 突发 QPS | '
        '突发下 RL 相对 SPARE | SPARE 容量下降 | RL 容量下降 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in data:
        lines.append(
            f'|{row["source"]}|{row["instantaneous_multiplier"]:g}x|'
            f'{row["spare_steady_qps"]:.3f} / {row["spare_burst_qps"]:.3f}|'
            f'{row["rl_steady_mean_qps"]:.3f} / {row["rl_burst_mean_qps"]:.3f}|'
            f'{row["rl_vs_spare_pct"]:+.2f}%|'
            f'{row["spare_drop_pct"]:.2f}%|{row["rl_drop_pct"]:.2f}%|')
    lines.extend(['',
                  '表中 RL 为三个训练种子的算术平均；逐种子值与范围见 '
                  '[CSV](capacity_burst.csv)。该实验只改变到达时间分布，'
                  '不改变请求长度组成或平均名义输入 QPS。'
                  '报告的是 CPU simulator 容量点估计，不是 GPU 实机吞吐量。',
                  '',
                  f'完成 {expected} 个组；所有通过/失败端点均已从原始 50 窗口记录复核。'
                  '紧邻失败点全部仅因 mean TTFT 超过 2 s，P99 TBT、积压与窗口有效性仍合格。'])
    (ANALYSIS / 'REPORT_CN.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
