"""Paired window summaries for the completed unified ablation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from research.www_component_ablation_mean_20261019 import behavior as run

READ, WRITE = run.READ, run.WRITE
OUT = run.OUT / 'analysis'
RESAMPLES = 2000
SEED = 20261020


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values, indices):
    boot = np.asarray(values)[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(boot, (.025, .975)))


def mean_ratio(rows, num, den):
    values = [r['counters'].get(num, 0)/r['counters'][den]
              for r in rows if r['counters'].get(den, 0)]
    return float(np.mean(values))*100 if values else None


def report(rows, audit):
    lookup = {(r['source'], r['kv_blocks'], r['arm'], r['rate_qps']): r for r in rows}
    lines = ['# 统一消融：常规负载行为证据', '',
             'Code/Conversation 各 24 个配对窗口，KV=1024/2048/4096，'
             '九个主变体，预先固定输入率 1/2/3/4 QPS，共 216 组、5,184 次拟合时钟模拟回放。'
             '全部有效，未运行模型 GPU 推理。', '',
             f"采集时可找到旧归档的 {audit['archive_parity_runs']:,} 次回放均通过轨迹哈希、"
             'mean TTFT、P99 TBT、积压、全程抢占和重算的逐项相等检查。其余点为新增采集点，'
             '后续容量扫描如提供对应记录，再在最终审核中核对。', '',
             '行为指标统计 [20,60) s 内开始的决策轮；延迟统计该时段到达的请求，并追踪至完成。'
             '冲突率先在各窗口除以具有该输出机会的轮数，再等权平均；零分母窗口排除并保留数量。'
             '潜在冲突要求候选满足省块及其他有效保护，决策改变则在相同状态重新按原顺序选策。', '',
             '## 完成保护：共同 2 QPS', '',
             '|负载|KV|Full 潜在冲突轮|Full 切换保护改变计划轮|关保护实际冲突轮|关保护冲突率 (%)|关保护−Full mean TTFT (ms)|95% 配对区间 (ms)|',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for source in run.pilot.SOURCES:
        for kv in run.pilot.KVS:
            f = lookup[source, kv, 'spare', 2.]
            n = lookup[source, kv, 'no_finish_guard', 2.]
            lines.append(f"|{source}|{kv}|{f['finish_potential_rounds_total']}|"
                         f"{f['finish_toggle_changed_rounds_total']}|{n['finish_conflict_rounds_total']}|"
                         f"{n['finish_conflict_pct']:.3f}|{n['delta_mean_ttft_vs_full_ms']:+.3f}|"
                         f"[{n['delta_mean_ttft_low_ms']:.3f}, {n['delta_mean_ttft_high_ms']:.3f}]|")
    lines += ['', '前三个计数列跨 24 窗口求和；冲突率是窗口比例的均值，不能用这三个总计数互相相除代替。'
              'Full 在所有已采集条件下实际完成冲突均为零。两策略闭环状态不同，Full 的同状态切换计数'
              '不等于关保护轨迹的实际冲突总数。保护减少的是可行基础计划中输出机会的削减；'
              'recovery 阶段的 KV 抢占另计。', '',
              '## 候选动作：共同 2 QPS', '',
              '|负载|KV|Full 选择候选 1/2/3/4 的轮数|去候选 1−Full mean TTFT (ms)|95% 配对区间 (ms)|Full/去候选 1 重算 (k tokens/窗口)|',
              '|---|---:|---|---:|---|---|']
    for source in run.pilot.SOURCES:
        for kv in run.pilot.KVS:
            f = lookup[source, kv, 'spare', 2.]
            n = lookup[source, kv, 'drop_candidate1', 2.]
            selected = '/'.join(str(f[f'candidate{i}_selected_total']) for i in range(1, 5))
            lines.append(f"|{source}|{kv}|{selected}|{n['delta_mean_ttft_vs_full_ms']:+.3f}|"
                         f"[{n['delta_mean_ttft_low_ms']:.3f}, {n['delta_mean_ttft_high_ms']:.3f}]|"
                         f"{f['recomputed_tokens_mean']/1000:.2f}/{n['recomputed_tokens_mean']/1000:.2f}|")
    lines += ['', '候选 1=(256,256)，2=(512,256)，3=(1024,64)，4=(512,64)，按该顺序选择。'
              '选择轮数跨窗口求和；重算为测量期每窗口均值。KV=1024 时删除候选 1 后两个负载的'
              '起答延迟和重算均增大；KV=4096、2 QPS 下 Full 没有选择暂缓候选，删除候选 1 也未改变轨迹。'
              '这支持以候选使用与资源压力解释差异，不能由 Code/Conversation 名称直接推断原因。', '',
              '单项删除也可能触发候选替代。Code/1024、2 QPS 下，Full 选择候选 3 共 307 次、'
              '候选 4 为零次；删除候选 3 后，候选 4 被选择 307 次，24 条执行轨迹均保持相同。'
              '因此零容量差不能直接解释为 chunk 候选没有作用。完整四档 QPS 下，删除候选 3/4 '
              '分别改变 2/10 条配对轨迹；已测容量相同也不等于所有动作或延迟完全相同。', '',
              '## 动态门槛：共同 2 QPS', '',
              '|负载|KV|Full/n=1 暂缓轮数|n=1−Full mean TTFT (ms)|95% 配对区间 (ms)|',
              '|---|---:|---|---:|---|']
    for source in run.pilot.SOURCES:
        for kv in run.pilot.KVS:
            f = lookup[source, kv, 'spare', 2.]
            n = lookup[source, kv, 'constant1', 2.]
            fc = sum(f[f'candidate{i}_selected_total'] for i in range(1, 5))
            nc = sum(n[f'candidate{i}_selected_total'] for i in range(1, 5))
            lines.append(f"|{source}|{kv}|{fc}/{nc}|{n['delta_mean_ttft_vs_full_ms']:+.3f}|"
                         f"[{n['delta_mean_ttft_low_ms']:.3f}, {n['delta_mean_ttft_high_ms']:.3f}]|")
    lines += ['', 'n=1 保留正物理省块要求、候选和保护。动态门槛在较宽松配置减少了暂缓；'
              '较紧张的 Conversation/1024 下，n=1 的更积极暂缓又改善了 mean TTFT。'
              '这是一组条件性取舍；容量与同 QPS 延迟须分别解释。', '',
              '## Decode 保护与完整数据', '',
              '九变体、全部四档 QPS 未观察到实际 decode 配额冲突；去 decode 保护与 Full 的'
              '24 个条件 × 24 窗口=576 对轨迹全部相同。当前排序优先正常 decode，'
              '最小候选预算 256 大于并发上限 32，候选 chunk 也足以保留单 token decode。'
              '该结构与零冲突结果相符，不推广为其他候选或排序下保护均冗余。', '',
              '完整数值：[behavior_summary.csv](behavior_summary.csv)；'
              '验证：[audit.json](audit.json)；采样及配置：`../design.json`；'
              '逐窗口指标、计数与最多八个切换事件：`../rates/`。', '',
              '配对 mean TTFT、抢占与重算区间按窗口重采样 2,000 次；'
              '行为计数不作为独立请求或独立实验样本。容量统计使用对应的 r4 容量扫描。', '']
    (OUT / 'REPORT_CN.md').write_text('\n'.join(lines))


def analyze(write_report=True):
    status = READ(run.OUT / 'status.json')
    expected_groups = len(run.ARMS)*len(run.pilot.SOURCES)*len(run.pilot.KVS)*len(run.RATES)
    assert status['completed'] == status['total'] == expected_groups and not status['errors']
    design = READ(run.OUT / 'design.json')
    rows_out, audit = [], dict(runs=0, archive_parity_runs=0, validity_errors=0,
                              protected_finish_conflicts=0, protected_decode_conflicts=0)
    rng = np.random.default_rng(SEED)
    for source in run.pilot.SOURCES:
        names = sorted(s['name'] for s in design['samples'] if s['source'] == source)
        assert len(names) == 24
        indices = rng.integers(0, 24, (RESAMPLES, 24))
        for kv in run.pilot.KVS:
            for rate in sorted(run.RATES):
                def read_arm(arm):
                    path = run.OUT / 'rates' / run.pilot.old.rate_name(source, kv, arm, rate)
                    raw = READ(path)['rows']
                    assert len(raw) == 24 and all(r['valid'] for r in raw)
                    by_name = {r['name']: r for r in raw}
                    assert set(by_name) == set(names)
                    return [by_name[name] for name in names]

                full = read_arm('spare')
                for arm in run.ARMS:
                    rows = read_arm(arm)
                    assert [r['measured_requests'] for r in rows] == [r['measured_requests'] for r in full]
                    delta = [r['mean_ttft_ms']-f['mean_ttft_ms'] for r, f in zip(rows, full)]
                    low, high = interval(delta, indices)
                    value = dict(source=source, kv_blocks=kv, arm=arm, rate_qps=rate, windows=24,
                                 mean_ttft_ms=float(np.mean([r['mean_ttft_ms'] for r in rows])),
                                 delta_mean_ttft_vs_full_ms=float(np.mean(delta)),
                                 delta_mean_ttft_low_ms=low, delta_mean_ttft_high_ms=high,
                                 p99_tbt_window_mean_ms=float(np.mean([r['p99_tbt_ms'] for r in rows])),
                                 backlog_drift_mean=float(np.mean([r['backlog_drift'] for r in rows])),
                                 different_trajectories=sum(r['trajectory_sha256'] != f['trajectory_sha256']
                                                            for r, f in zip(rows, full)))
                    value['passes_2s'] = (value['mean_ttft_ms'] <= 2000 and
                                         value['p99_tbt_window_mean_ms'] <= 200 and
                                         value['backlog_drift_mean'] <= max(2, rate))
                    for kind in ('finish', 'decode'):
                        for what in ('eligible_rounds', 'eligible_instances', 'conflict_rounds',
                                     'conflict_instances', 'potential_rounds', 'toggle_changed_rounds'):
                            value[f'{kind}_{what}_total'] = sum(r['counters'].get(f'{kind}_{what}', 0)
                                                               for r in rows)
                        value[f'{kind}_eligible_windows'] = sum(bool(r['counters'].get(f'{kind}_eligible_rounds', 0))
                                                               for r in rows)
                        value[f'{kind}_conflict_pct'] = mean_ratio(rows, f'{kind}_conflict_rounds',
                                                                 f'{kind}_eligible_rounds')
                        value[f'{kind}_potential_pct'] = (None if arm == 'srp' else mean_ratio(
                            rows, f'{kind}_potential_rounds', f'{kind}_eligible_rounds'))
                        value[f'{kind}_toggle_changed_pct_all_rounds'] = (None if arm == 'srp' else mean_ratio(
                            rows, f'{kind}_toggle_changed_rounds', 'rounds'))
                    for key in ('rounds', 'recovery_rounds', 'preemptions', 'recomputed_tokens',
                                'intervention_rounds', 'deferred_tokens', 'deferred_request_rounds', 'saved_blocks'):
                        value[f'{key}_mean'] = float(np.mean([r['counters'].get(key, 0) for r in rows]))
                    for key in ('preemptions', 'recomputed_tokens', 'saved_blocks'):
                        differences = [r['counters'].get(key, 0)-f['counters'].get(key, 0)
                                       for r, f in zip(rows, full)]
                        lo, hi = interval(differences, indices)
                        value[f'delta_{key}_mean'] = float(np.mean(differences))
                        value[f'delta_{key}_low'] = lo
                        value[f'delta_{key}_high'] = hi
                    value['intervention_pct'] = mean_ratio(rows, 'intervention_rounds', 'rounds')
                    value['free_blocks_round_mean'] = sum(r['counters']['free_blocks_sum']/r['counters']['rounds'] for r in rows)/24
                    value['pending_prefill_tokens_round_mean'] = sum(r['counters']['pending_prefill_tokens_sum']/r['counters']['rounds'] for r in rows)/24
                    for i in range(1, 5):
                        key = f'selected_candidate{i}_rounds'
                        value[f'candidate{i}_selected_total'] = sum(r['counters'].get(key, 0) for r in rows)
                        value[f'candidate{i}_selected_pct_all_rounds'] = mean_ratio(rows, key, 'rounds')
                    for key in ('request_prompt_mean', 'request_output_mean'):
                        value[key] = float(np.mean([r[key] for r in rows]))
                    audit['runs'] += len(rows)
                    audit['archive_parity_runs'] += sum(bool(r['parity_sources']) for r in rows)
                    if arm != 'no_finish_guard':
                        audit['protected_finish_conflicts'] += value['finish_conflict_rounds_total']
                    if arm != 'no_decode_guard':
                        audit['protected_decode_conflicts'] += value['decode_conflict_rounds_total']
                    rows_out.append(value)
    assert audit['runs'] == expected_groups*24
    assert audit['protected_finish_conflicts'] == audit['protected_decode_conflicts'] == 0
    save(OUT / 'behavior_summary.csv', rows_out)
    WRITE(OUT / 'audit.json', audit)
    WRITE(OUT / 'analysis_design.json', dict(resamples=RESAMPLES, seed=SEED,
                                           paired_unit='source window',
                                           conflict_aggregation='mean of per-window ratios; zero denominators excluded'))
    if write_report:
        report(rows_out, audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    analyze()
