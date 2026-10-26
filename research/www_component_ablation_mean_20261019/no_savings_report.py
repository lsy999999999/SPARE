"""Audit and report the actual deletion of the savings acceptance condition."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from research.www_component_ablation_mean_20261019 import no_savings as run

READ, WRITE = run.READ, run.WRITE


def audit():
    for root, expected in ((run.OUT, 18), (run.BEHAVIOR_OUT, 72)):
        status = READ(root / 'status.json')
        assert status['completed'] == status['total'] == expected and not status['errors']
    old = Path(run.pilot.policy.__file__).read_text().split('\n', 1)[1]
    new = Path(run.no_savings_policy.__file__).read_text().split('\n', 1)[1]
    assert new == old.replace("elif d['acceptance_saving'] < n:",
                             "elif not opt.get('disable_saving_condition', False) and d['acceptance_saving'] < n:")
    design = READ(run.OUT / 'design.json')
    assert design['policy_sha256'] == run.DIGEST(Path(run.no_savings_policy.__file__))
    assert design['observer_sha256'] == run.DIGEST(Path(run.no_savings_observer.__file__))
    parent = READ(run.complete.OUT / 'design.json')
    assert design['samples'] == parent['samples']
    assert all(run.DIGEST(run.ROOT/path) == sha for path, sha in parent['input_sha256'].items())
    points = list(csv.DictReader((run.OUT / 'analysis/points.csv').open()))
    assert len(points) == 72
    boundaries = 0
    for p in points:
        for field in ('capacity_qps', 'next_fail_qps'):
            rate = float(p[field])
            if rate == 0:
                continue
            data = READ(run.OUT / 'rates' / run.pilot.old.rate_name(
                p['source'], int(p['kv_blocks']), p['arm'], rate))
            rows = data['rows']
            assert len(rows) == 24 and all(r['valid'] for r in rows)
            means = {k: sum(r[k] for r in rows)/24 for k in
                     ('mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift')}
            assert all(math.isclose(v, data['summary'][k], abs_tol=1e-8) for k, v in means.items())
            passed = (means['mean_ttft_ms'] <= float(p['theta_s'])*1000 and
                      means['p99_tbt_ms'] <= 200 and means['backlog_drift'] <= max(2, rate))
            assert passed == (field == 'capacity_qps')
            boundaries += 1
    reused = new_runs = 0
    for path in (run.OUT / 'rates').glob('*.json'):
        d = READ(path)
        new_runs += d['new_simulations']
        if d['reused_from']:
            prior = run.ROOT / d['reused_from']
            assert run.DIGEST(prior) == d['reused_sha256']
            assert d['rows'] == READ(prior)['rows']
            reused += len(d['rows'])
    behavior_reused = behavior_new = matched = main_matched = 0
    counters = dict(zero=0, negative=0, below_dynamic=0, positive=0)
    for path in (run.BEHAVIOR_OUT / 'rates').glob('*.json'):
        d = READ(path)
        if 'reused_from' in d:
            old_path = run.ROOT / d['reused_from']
            assert run.DIGEST(old_path) == d['reused_sha256']
            assert d['rows'] == READ(old_path)['rows']
            behavior_reused += len(d['rows'])
        else:
            behavior_new += len(d['rows'])
        prior_path = run.OUT / 'rates' / path.name
        if prior_path.exists():
            prior = {r['name']: r for r in READ(prior_path)['rows']}
            for r in d['rows']:
                for key in ('trajectory_sha256', 'mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift'):
                    assert r[key] == prior[r['name']][key]
                matched += 1
                main_matched += r['rate'] == 2
        for r in d['rows']:
            if r['arm'] == 'no_savings_condition':
                c = r['counters']
                for name, key in (('zero', 'selected_zero_saving_rounds'),
                                  ('negative', 'selected_negative_saving_rounds'),
                                  ('positive', 'selected_positive_saving_rounds'),
                                  ('below_dynamic', 'selected_below_dynamic_threshold_rounds')):
                    counters[name] += c.get(key, 0)
                assert c.get('finish_conflict_rounds', 0) == c.get('decode_conflict_rounds', 0) == 0
    assert counters['zero'] > 0
    assert behavior_new == 576 and behavior_reused == 1152
    assert main_matched == 3*6*24
    result = dict(capacity_groups=18, new_arm_groups=6, capacity_points=72,
                  boundary_endpoints_checked=boundaries, new_capacity_window_simulations=new_runs,
                  reused_capacity_window_records=reused, new_behavior_replays=behavior_new,
                  reused_behavior_replays=behavior_reused, behavior_capacity_parity=matched,
                  main_2qps_parity=main_matched, actual_new_arm_selection_counts=counters,
                  sole_policy_change_verified=True, input_pairing_verified=True, errors=[])
    WRITE(run.OUT / 'analysis/final_audit.json', result)
    return result


def report():
    result = audit()
    points = list(csv.DictReader((run.OUT / 'analysis/points.csv').open()))
    cap = {(r['source'], int(r['kv_blocks']), r['arm']): float(r['capacity_qps'])
           for r in points if float(r['theta_s']) == 2}
    stats = list(csv.DictReader((run.BEHAVIOR_OUT / 'analysis/behavior_summary.csv').open()))
    latency = {(r['source'], int(r['kv_blocks']), r['arm']): r for r in stats if float(r['rate_qps']) == 2}
    intervals = {(r['source'], int(r['kv_blocks']), r['arm']): r for r in csv.DictReader(
        (run.OUT / 'analysis/paired_capacity_bootstrap_2s.csv').open())}
    lines = ['# 完全删除省块接受条件：24 窗口配对结果', '',
             '本实验直接跳过 `acceptance_saving >= n_t` 判断。零省块和负省块候选都不会'
             '因省块条件被拒绝；没有以 n=1、n=0 或极小常数代替删除。候选顺序、物理可行性、'
             '完成/decode 保护、基础计划有正新增 KV 分配及至少两个请求的触发条件保持原样。', '',
             '原 n=1 实验保留为参数敏感性对照；Full 与 n=1 的容量和行为记录直接复用原 24 窗口归档。'
             '新变体使用相同 Code/Conversation 窗口、到达序列、KV、时钟和测量区间。'
             '以下容量为 mean TTFT≤2 s、窗口 P99 TBT 均值≤200 ms、积压约束下的输入 QPS；'
             '延迟在共同 2 QPS 下测量。全部为 CPU 拟合时钟模拟。', '',
             '## Full 与真正去省块条件的比较', '',
             '|负载|KV|Full 容量|去省块条件容量|容量差及 95% 区间 (QPS)|Full mean TTFT (ms)|去省块条件 mean TTFT (ms)|延迟差及 95% 区间 (ms)|',
             '|---|---:|---:|---:|---|---:|---:|---|']
    for source in run.pilot.SOURCES:
        for kv in run.pilot.KVS:
            f, n = latency[source, kv, 'spare'], latency[source, kv, 'no_savings_condition']
            c = intervals[source, kv, 'no_savings_condition']
            lines.append(f"|{source}|{kv}|{cap[source,kv,'spare']:.2f}|{cap[source,kv,'no_savings_condition']:.2f}|"
                         f"{float(c['delta_capacity_qps']):+.2f} [{float(c['bootstrap_low_qps']):.2f}, {float(c['bootstrap_high_qps']):.2f}]|"
                         f"{float(f['mean_ttft_ms']):.2f}|{float(n['mean_ttft_ms']):.2f}|"
                         f"{float(n['delta_mean_ttft_vs_full_ms']):+.2f} [{float(n['delta_mean_ttft_low_ms']):.2f}, {float(n['delta_mean_ttft_high_ms']):.2f}]|")
    lines += ['', '所有差值均为去省块条件减 Full。容量点估计和同 QPS 延迟分别解释；'
              '区间按 24 个配对窗口重采样 2,000 次，不能将离散扫描精度当作置信精度。', '',
              '在共同 2 QPS 下，删除条件后六组 mean TTFT 均更低，其配对 95% 区间均不跨零。'
              '2 s 容量点估计三组提高、三组降低，六组容量差区间均跨零。因此当前证据呈现容量与'
              '起答延迟的条件性取舍，不支持该省块条件在已测设置中具有一致性能增益。', '',
              '## 与 n=1 的区别及实际接受行为', '',
              '|负载|KV|n=1 容量|n=1 mean TTFT (ms)|去条件/Full 轨迹不同窗口数|去条件/n=1 轨迹不同窗口数|去条件选中零省块/负省块轮数|零省块但实际改变计划轮数|',
              '|---|---:|---:|---:|---:|---:|---|---:|']
    for source in run.pilot.SOURCES:
        for kv in run.pilot.KVS:
            names = lambda arm: {r['name']: r for r in READ(run.BEHAVIOR_OUT / 'rates' /
                run.pilot.old.rate_name(source, kv, arm, 2.))['rows']}
            new, full, one = names('no_savings_condition'), names('spare'), names('constant1')
            zeros = sum(r['counters'].get('selected_zero_saving_rounds', 0) for r in new.values())
            negatives = sum(r['counters'].get('selected_negative_saving_rounds', 0) for r in new.values())
            noops = sum(r['counters'].get('selected_noop_rounds', 0) for r in new.values())
            n_one = sum(new[name]['trajectory_sha256'] != one[name]['trajectory_sha256'] for name in new)
            n_full = sum(new[name]['trajectory_sha256'] != full[name]['trajectory_sha256'] for name in new)
            lines.append(f"|{source}|{kv}|{cap[source,kv,'constant1']:.2f}|"
                         f"{float(latency[source,kv,'constant1']['mean_ttft_ms']):.2f}|{n_full}|{n_one}|"
                         f"{zeros}/{negatives}|{zeros-noops}|")
    lines += ['', '计数跨 24 窗口的 [20,60) s 决策轮求和。零省块候选有时与基础计划相同，'
              '有时实际减少 token 但不减少物理块；表中将两者区分。零省块候选提前被接受还可能'
              '改变后续候选的选择机会，因此 n=1 和删除条件不能互换。', '',
              '## 验证与复现', '',
              f"新增容量窗口模拟 {result['new_capacity_window_simulations']:,} 次，复用 {result['reused_capacity_window_records']:,} 条容量窗口记录；"
              '新增行为回放 576 次，复用 1,152 次。18 组容量和 72 组行为均完成且零错误。'
              f"核对 {result['boundary_endpoints_checked']} 个通过/失败边界、冻结输入、单项代码改动与 {result['main_2qps_parity']} 条共同 2 QPS 配对回放。", '',
              '单元检查明确验证零省块与负省块不再被拒绝，同时保护和可行性检查仍有效。'
              f"四档 QPS 的实际记录中，新变体选中零省块 {result['actual_new_arm_selection_counts']['zero']:,} 轮，"
              f"负省块 {result['actual_new_arm_selection_counts']['negative']:,} 轮，确认实际运行也接受了非正省块候选。", '',
              '复现：`.vllm-venv/bin/python -m research.www_component_ablation_mean_20261019.check_no_savings`；'
              '运行 `research.www_component_ablation_mean_20261019.no_savings` 的 `--smoke`、`--workers 12`、`--analyze`；'
              '最后运行 `research.www_component_ablation_mean_20261019.no_savings_report`。', '',
              '原始容量：`../rates/`；容量及区间：[points.csv](points.csv)、[paired_capacity_bootstrap_2s.csv](paired_capacity_bootstrap_2s.csv)；'
              '行为及配对延迟：[behavior_summary.csv](../behavior/analysis/behavior_summary.csv)；'
              '验收：[final_audit.json](final_audit.json)。', '']
    (run.OUT / 'analysis/REPORT_CN.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    report()
