"""Build the three-part ablation table from matched 24-window archives."""
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
CAPACITY = ROOT / 'artifacts/www_component_ablation_mean_20261020_r4'
BEHAVIOR = ROOT / 'artifacts/www_component_behavior_20261020_r1'
ARRIVAL = ROOT / 'artifacts/www_arrival_behavior_20261020_r1'
ARRIVAL_REVIEW = ROOT / 'artifacts/www_arrival_term_review_20261020_r1'
OUT = ROOT / 'writing/www/manuscript/ablation_20261020'
ARMS = ('spare', 'drop_candidate1', 'drop_candidate2', 'drop_candidate3',
        'drop_candidate4', 'no_finish_guard', 'no_decode_guard', 'no_arrival')
MAIN = ('spare', 'drop_candidate1', 'no_finish_guard', 'no_decode_guard', 'no_arrival')
LABELS = dict(spare='Full SPARE', drop_candidate1='w/o candidate 1',
              drop_candidate2='w/o candidate 2', drop_candidate3='w/o candidate 3',
              drop_candidate4='w/o candidate 4', no_finish_guard='w/o finish guard',
              no_decode_guard='w/o decode guard', no_arrival='w/o arrival term')


def read(path):
    return json.loads(path.read_text())


def main():
    design = read(CAPACITY / 'design.json')
    behavior_design = read(BEHAVIOR / 'design.json')
    arrival_design = read(ARRIVAL / 'design.json')
    assert design['samples'] == behavior_design['samples'] == arrival_design['samples']
    assert read(ARRIVAL / 'status.json')['completed'] == 6
    points = {(r['source'], int(r['kv_blocks']), r['arm']): r
              for r in csv.DictReader((CAPACITY / 'analysis/points.csv').open())
              if float(r['theta_s']) == 2}
    standard = {(r['source'], int(r['kv_blocks']), r['arm']): r
                for r in csv.DictReader((BEHAVIOR / 'analysis/behavior_summary.csv').open())
                if float(r['rate_qps']) == 2}
    arrival_intervals = {(r['source'], int(r['kv_blocks'])): r
                         for r in csv.DictReader((ARRIVAL_REVIEW / 'paired_24windows.csv').open())}
    capacity_intervals = {(r['source'], int(r['kv_blocks']), r['arm']): r
                          for r in csv.DictReader((CAPACITY / 'analysis/paired_capacity_bootstrap_2s.csv').open())}
    rows = []
    for source in ('code', 'conversation'):
        names = sorted(s['name'] for s in design['samples'] if s['source'] == source)
        assert len(names) == 24
        for kv in (1024, 2048, 4096):
            for arm in ARMS:
                raw = read(CAPACITY / 'rates' / f'{source}_c{kv}_{arm}_r2.json')['rows']
                assert sorted(r['name'] for r in raw) == names
                assert all(r['valid'] for r in raw)
                ttft = mean(r['mean_ttft_ms'] for r in raw)
                if arm == 'no_arrival':
                    observed = read(ARRIVAL / 'rates' /
                                    f'{source}_c{kv}_no_arrival_r2.json')['rows']
                    assert sorted(r['name'] for r in observed) == names
                    assert all(r['parity_sources'] for r in observed)
                    assert {r['name']: r['trajectory_sha256'] for r in raw} == {
                        r['name']: r['trajectory_sha256'] for r in observed}
                    assert abs(ttft-mean(r['mean_ttft_ms'] for r in observed)) < 1e-8
                    counters = [r['counters'] for r in observed]
                    conflict = mean(c.get('finish_conflict_rounds', 0) /
                                    c['finish_eligible_rounds'] for c in counters
                                    if c.get('finish_eligible_rounds', 0))*100
                    conflict_rounds = sum(c.get('finish_conflict_rounds', 0)
                                          for c in counters)
                    recomputed = mean(c['recomputed_tokens'] for c in counters)
                    selected1 = sum(c.get('selected_candidate1_rounds', 0) for c in counters)
                    behavior_source = ARRIVAL / 'rates' / f'{source}_c{kv}_no_arrival_r2.json'
                    interval = arrival_intervals[source, kv]
                    ttft_delta_low = float(interval['ttft_delta_low_ms'])
                    ttft_delta_high = float(interval['ttft_delta_high_ms'])
                else:
                    b = standard[source, kv, arm]
                    assert abs(ttft-float(b['mean_ttft_ms'])) < 1e-8
                    conflict = float(b['finish_conflict_pct'])
                    conflict_rounds = int(b['finish_conflict_rounds_total'])
                    recomputed = float(b['recomputed_tokens_mean'])
                    selected1 = int(b['candidate1_selected_total'])
                    behavior_source = BEHAVIOR / 'analysis/behavior_summary.csv'
                    ttft_delta_low = float(b['delta_mean_ttft_low_ms'])
                    ttft_delta_high = float(b['delta_mean_ttft_high_ms'])
                cap_interval = capacity_intervals.get((source, kv, arm))
                if arm == 'no_arrival':
                    capacity_delta_low = float(interval['capacity_delta_low_qps'])
                    capacity_delta_high = float(interval['capacity_delta_high_qps'])
                elif cap_interval:
                    capacity_delta_low = float(cap_interval['bootstrap_low_qps'])
                    capacity_delta_high = float(cap_interval['bootstrap_high_qps'])
                else:
                    assert arm == 'spare'
                    capacity_delta_low = capacity_delta_high = 0.
                row = dict(source=source, kv_blocks=kv, arm=arm, label=LABELS[arm],
                           windows=24, capacity_limit_mean_ttft_s=2,
                           capacity_qps=float(points[source, kv, arm]['capacity_qps']),
                           input_rate_for_behavior_qps=2, mean_ttft_ms=ttft,
                           finish_conflict_rounds=conflict_rounds,
                           finish_conflict_pct=conflict,
                           recomputed_k_tokens_per_window=recomputed/1000,
                           candidate1_selected_rounds=selected1,
                           ttft_delta_95ci_low_ms=ttft_delta_low,
                           ttft_delta_95ci_high_ms=ttft_delta_high,
                           capacity_delta_95ci_low_qps=capacity_delta_low,
                           capacity_delta_95ci_high_qps=capacity_delta_high,
                           capacity_source=str((CAPACITY / 'analysis/points.csv').relative_to(ROOT)),
                           latency_source=str((CAPACITY / 'rates' /
                                               f'{source}_c{kv}_{arm}_r2.json').relative_to(ROOT)),
                           behavior_source=str(behavior_source.relative_to(ROOT)))
                rows.append(row)
    assert len(rows) == 48
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'all_kv_24windows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    shown = [r for r in rows if r['kv_blocks'] == 1024 and r['arm'] in MAIN]
    shown.sort(key=lambda r: (('code', 'conversation').index(r['source']), MAIN.index(r['arm'])))
    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{Three SPARE ablations at 1,024 KV blocks. Each workload uses 24 paired '
             r'windows in the fitted-clock simulator. Capacity is the last passing scanned '
             r'input rate (QPS) before the first failure under the 2-s mean-TTFT limit, '
             r'the 200-ms mean window-P99 TBT '
             r'limit, and the backlog constraint. Mean TTFT, finish conflicts, and '
             r'recomputed tokens use a common 2-QPS input rate. Finish conflicts show '
             r'the total number of scheduling rounds across 24 windows and, in '
             r'parentheses, the mean per-window percentage of eligible rounds in which '
             r'the executed plan reduces a prefill allocation that could have completed '
             r'under the feasible base plan. The 2-QPS latency of Conversation without '
             r'candidate 1 is measured above the 1.76-QPS capacity of that variant.}',
             r'\label{tab:spare-three-part-ablation}', r'\small',
             r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{@{}llrrrr@{}}', r'\toprule',
             r'Workload & Variant & \shortstack{Capacity\\(QPS) $\uparrow$} '
             r'& \shortstack{Mean TTFT\\(ms) $\downarrow$} '
             r'& \shortstack{Finish conflicts\\rounds (\%) $\downarrow$} '
             r'& \shortstack{Recompute\\(k/window) $\downarrow$} \\',
             r'\midrule']
    for i, r in enumerate(shown):
        if i == 5:
            lines.append(r'\midrule')
        source = r['source'].capitalize() if r['arm'] == 'spare' else ''
        lines.append(f"{source} & {r['label']} & {r['capacity_qps']:.2f} & "
                     f"{r['mean_ttft_ms']:.0f} & "
                     f"{r['finish_conflict_rounds']} ({r['finish_conflict_pct']:.2f}) & "
                     f"{r['recomputed_k_tokens_per_window']:.2f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    (OUT / 'table.tex').write_text('\n'.join(lines))
    print(f'Wrote {len(rows)} matched rows and {len(shown)} main-table rows')


if __name__ == '__main__':
    main()
