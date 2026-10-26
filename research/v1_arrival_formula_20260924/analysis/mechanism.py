"""Post-run diagnosis of integer thresholds versus actual action changes."""
import gzip
import json
import math
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.v1_arrival_formula_20260924.common import OUT, read, write


def diagnose(result):
    with gzip.open(OUT / result['rounds_path'], 'rt') as stream:
        rounds = json.load(stream)
    eligible = threshold_changed = action_changed = first_second = first_second_changed = 0
    old_ns = []
    new_ns = []
    first_ns = []
    first_old_ns = []
    ratios = []
    for row in rounds:
        if row['baseline'] is None or row['baseline_growth_blocks'] <= 0 or len(row['state']) < 2:
            continue
        candidates = [c for c in row['candidate_diagnostics']
                      if c['tokens'] is not None and c['guard'] and c['saving'] >= 1]
        if not candidates:
            continue
        old = max(1, min(100000, math.ceil(8 + 128 * row['features']['work_free_ratio'])))
        new = row['threshold']
        def action(n):
            return next((c['tokens'] for c in candidates if c['saving'] >= n), row['baseline'])
        assert action(new) == row['scheduled']
        eligible += 1
        threshold_changed += old != new
        action_changed += action(old) != action(new)
        old_ns.append(old)
        new_ns.append(new)
        w = sum(max(0, s['prompt'] - s['computed']) for s in row['state'] if s['outputs'] == 0)
        ratios.append(row['features']['arrival_rate'] / (16 + w))
        if row['features']['arrival_time_s'] <= 1:
            first_second += 1
            first_second_changed += action(old) != action(new)
            first_ns.append(new)
            first_old_ns.append(old)
    return dict(key=result['job']['key'], case=result['job']['case'], capacity=result['job']['capacity'],
                repeat=result['job']['repeat'], eligible=eligible, threshold_changed=threshold_changed,
                action_changed=action_changed, old_n_range=[min(old_ns), max(old_ns)] if old_ns else [],
                new_n_range=[min(new_ns), max(new_ns)] if new_ns else [],
                first_second_opportunities=first_second, first_second_actions_changed=first_second_changed,
                first_second_old_n_range=[min(first_old_ns),max(first_old_ns)] if first_old_ns else [],
                first_second_new_n_range=[min(first_ns),max(first_ns)] if first_ns else [],
                arrival_to_backlog_ratio_median=st.median(ratios) if ratios else None)


if __name__ == '__main__':
    assert read(OUT / 'status.json')['state'] == 'complete'
    results = [read(p) for p in sorted((OUT / 'runs').glob('*.json'))]
    rows = [diagnose(r) for r in results if r['job'].get('arm') == 'arrival']
    assert len(rows) == 24
    write(OUT / 'analysis/mechanism.json', rows)
    for row in rows:
        print(json.dumps(row))
