"""Assemble existing paired ablations without mixing window cohorts."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'writing/www/figures/ablation_unified'
ARMS = (
    ('reference', 'spare', 'Full'),
    ('reference', 'srp', 'SRP / 无暂缓'),
    ('threshold', 'no_savings_condition', '去省块条件'),
    ('candidate', 'drop_candidate1', '去候选 1'),
    ('candidate', 'drop_candidate2', '去候选 2'),
    ('candidate', 'drop_candidate3', '去候选 3'),
    ('candidate', 'drop_candidate4', '去候选 4'),
    ('protection', 'no_finish_guard', '去完成保护'),
    ('protection', 'no_decode_guard', '去 decode 保护'),
)


def build(windows, suffix):
    root = ROOT / (f'artifacts/www_component_ablation_mean_20261019_{suffix}' if windows == 12
                   else 'artifacts/www_component_ablation_mean_20261020_r4')
    points = list(csv.DictReader((root / 'analysis/points.csv').open()))
    lookup = {(r['source'], int(r['kv_blocks']), r['arm']): r
              for r in points if float(r['theta_s']) == 2}
    behaviors, intervals = {}, {}
    no_savings_root = ROOT / 'artifacts/www_no_savings_condition_20261020_r1'
    behavior_source = ROOT / 'artifacts/www_component_behavior_20261020_r1/analysis/behavior_summary.csv'
    no_savings_behavior_source = no_savings_root / 'behavior/analysis/behavior_summary.csv'
    if windows == 24:
        behaviors = {(r['source'], int(r['kv_blocks']), r['arm']): r
                     for r in csv.DictReader(behavior_source.open()) if float(r['rate_qps']) == 2}
        intervals = {(r['source'], int(r['kv_blocks']), r['arm']): r for r in csv.DictReader(
            (root / 'analysis/paired_capacity_bootstrap_2s.csv').open())}
        lookup.update({(r['source'], int(r['kv_blocks']), r['arm']): r
                       for r in csv.DictReader((no_savings_root / 'analysis/points.csv').open())
                       if r['arm'] == 'no_savings_condition' and float(r['theta_s']) == 2})
        behaviors.update({(r['source'], int(r['kv_blocks']), r['arm']): r
                          for r in csv.DictReader(no_savings_behavior_source.open())
                          if r['arm'] == 'no_savings_condition' and float(r['rate_qps']) == 2})
        intervals.update({(r['source'], int(r['kv_blocks']), r['arm']): r
                          for r in csv.DictReader((no_savings_root / 'analysis/paired_capacity_bootstrap_2s.csv').open())
                          if r['arm'] == 'no_savings_condition'})
    output = []
    lines = [f'# 统一消融表：每类 {windows} 个配对窗口', '',
             '容量：mean TTFT 上限 2 s、窗口 P99 TBT 均值 ≤200 ms、'
             '平均积压变化 ≤max(2,QPS)。mean TTFT：共同输入率 2 QPS，先逐窗口计算再取均值。', '',
             '所有数值来自同一窗口集合。缺失格表示未完成该窗口数的对应实验；'
             '“未采集”表示原始归档缺少保护冲突计数，不表示零。容量为离散扫描点估计。', '']
    if windows == 24:
        lines[4] = ('全部容量、延迟与行为指标使用相同的 24 个配对窗口。冲突率为逐窗口'
                    '“实际削减保护配额的轮数 / 有该保护机会的轮数”的均值；零分母窗口排除。'
                    'KV 抢占和重算为 [20,60) s 决策轮的每窗口均值。所有行为指标均在共同 2 QPS 测量。')
        lines += ['容量是扫描点估计，不同变体的容量点具有不同输入率。'
                  '配对容量差和 mean TTFT 差的 95% bootstrap 区间保存在 CSV；'
                  '0.01–0.02 QPS 是扫描精度。下表的冲突率按百分比显示，重算按千 token 显示。', '']
        lines += ['“去省块条件”直接跳过整个省块接受判断，允许零省块或负省块候选；'
                  '可行性、保护、候选顺序和其他触发条件保持不变。n=1 已移至参数敏感性比较。', '']
    for source in ('code', 'conversation'):
        for kv in (1024, 2048, 4096):
            lines += [f'## {source} / KV={kv}', '',
                      '| 方面 | 变体 | 容量 (QPS) | mean TTFT @2 QPS (ms) | 完成冲突 (%) | decode 冲突 (%) | KV 抢占/窗口 | 重算 (k tokens/窗口) |',
                      '|---|---|---:|---:|---:|---:|---:|---:|']
            for category, arm, label in ARMS:
                if windows == 12 and arm == 'no_savings_condition':
                    category, arm, label = 'sensitivity', 'constant1', '固定门槛 n=1（敏感性）'
                point = lookup.get((source, kv, arm))
                arm_root = no_savings_root if arm == 'no_savings_condition' else root
                arm_behavior_source = no_savings_behavior_source if arm == 'no_savings_condition' else behavior_source
                file = arm_root / 'rates' / f'{source}_c{kv}_{arm}_r2.json'
                value = None
                if file.exists():
                    raw = json.loads(file.read_text())['rows']
                    assert len(raw) == windows and len({r['name'] for r in raw}) == windows
                    assert all(r['valid'] for r in raw)
                    value = sum(r['mean_ttft_ms'] for r in raw) / windows
                capacity = float(point['capacity_qps']) if point else None
                key = source, kv, arm
                behavior = behaviors.get(key)
                ci = intervals.get(key)
                if windows == 24:
                    assert point is not None and behavior is not None and value is not None, key
                    assert abs(float(behavior['mean_ttft_ms'])-value) < 1e-8, key
                finish = float(behavior['finish_conflict_pct']) if behavior and behavior['finish_conflict_pct'] else None
                decode = float(behavior['decode_conflict_pct']) if behavior and behavior['decode_conflict_pct'] else None
                preemptions = float(behavior['preemptions_mean']) if behavior else None
                recomputation = float(behavior['recomputed_tokens_mean']) if behavior else None
                output.append(dict(source=source, kv_blocks=kv, windows=windows,
                                   category=category, arm=arm, label=label,
                                   capacity_qps=capacity, mean_ttft_at_2qps_ms=value,
                                   finish_conflict_pct=finish, decode_conflict_pct=decode,
                                   measured_preemptions_mean=preemptions,
                                   measured_recomputed_tokens_mean=recomputation,
                                   capacity_delta_vs_full_qps=ci['delta_capacity_qps'] if ci else (0 if arm == 'spare' else None),
                                   capacity_delta_low_qps=ci['bootstrap_low_qps'] if ci else (0 if arm == 'spare' else None),
                                   capacity_delta_high_qps=ci['bootstrap_high_qps'] if ci else (0 if arm == 'spare' else None),
                                   ttft_delta_vs_full_ms=behavior['delta_mean_ttft_vs_full_ms'] if behavior else None,
                                   ttft_delta_low_ms=behavior['delta_mean_ttft_low_ms'] if behavior else None,
                                   ttft_delta_high_ms=behavior['delta_mean_ttft_high_ms'] if behavior else None,
                                   conflict_status='measured' if behavior else 'not_collected',
                                   behavior_source=str(arm_behavior_source.relative_to(ROOT)) if behavior else '',
                                   capacity_source=str(arm_root.relative_to(ROOT) / 'analysis/points.csv'),
                                   latency_source=str(file.relative_to(ROOT)) if file.exists() else ''))
                cap_text = f'{capacity:.2f}' if capacity is not None else '待补齐'
                ttft_text = f'{value:.2f}' if value is not None else '待补齐'
                def fmt(number, digits=3):
                    return f'{number:.{digits}f}' if number is not None else '未采集'
                category_text = dict(reference='参照', threshold='门槛', candidate='候选', protection='保护', sensitivity='敏感性')[category]
                lines.append(f'| {category_text} | {label} | {cap_text} | {ttft_text} | '
                             f'{fmt(finish)} | {fmt(decode)} | {fmt(preemptions, 2)} | '
                             f'{fmt(recomputation/1000 if recomputation is not None else None, 2)} |')
            lines.append('')
    target = OUT / f'unified_{windows}windows'
    target.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    with target.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return len(output), sum(r['capacity_qps'] is None for r in output)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for windows, suffix in ((24, 'r3'), (12, 'r2')):
        rows, missing = build(windows, suffix)
        print(f'{windows} windows: {rows} rows, {missing} capacity entries pending')
