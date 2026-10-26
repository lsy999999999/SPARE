"""Compare archived controls on exactly the no-savings arm's 24 windows.

Descriptive diagnostic at 2 QPS, not a new capacity scan or GPU validation.
"""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / 'artifacts/www_no_savings_condition_20261020_r1'
OLD = ROOT / 'artifacts/www_biobjective_ablation_mean_20261019_r2'


def read(path):
    return json.loads(path.read_text())


def main():
    new_design, old_design = read(NEW / 'design.json'), read(OLD / 'design.json')
    prior_samples = {s['name']: s for s in old_design['samples']}
    assert all(prior_samples[s['name']] == s for s in new_design['samples'])
    behavior = list(csv.DictReader((NEW / 'behavior/analysis/behavior_summary.csv').open()))
    rows, sources = [], {}
    for source in ('code', 'conversation'):
        wanted = {s['name'] for s in new_design['samples'] if s['source'] == source}
        assert len(wanted) == 24
        for kv in (1024, 2048, 4096):
            row = dict(source=source, kv_blocks=kv, qps=2, windows=24)
            for arm in ('spare', 'no_savings_condition', 'fixed256', 'fixed512', 'srp'):
                root = NEW if arm in ('spare', 'no_savings_condition') else OLD
                path = root / 'rates' / f'{source}_c{kv}_{arm}_r2.json'
                matched = [r for r in read(path)['rows'] if r['name'] in wanted]
                assert len(matched) == 24 and {r['name'] for r in matched} == wanted
                assert all(r['valid'] and r['capacity'] == kv and r['rate'] == 2 for r in matched)
                row[f'{arm}_mean_ttft_ms'] = mean(r['mean_ttft_ms'] for r in matched)
                sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            b = next(r for r in behavior if r['source'] == source and int(r['kv_blocks']) == kv
                     and r['arm'] == 'no_savings_condition' and float(r['rate_qps']) == 2)
            counts = [int(b[f'candidate{i}_selected_total']) for i in range(1, 5)]
            row['candidate1_pct_of_selected_candidates'] = 100 * counts[0] / sum(counts)
            rows.append(row)
    output = NEW / 'analysis'
    with (output / 'small_budget_controls_2qps.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / 'small_budget_controls_audit.json').write_text(json.dumps(dict(
        rows=rows, sources_sha256=sources, new_simulations=0,
        interpretation='Matched-window descriptive means only; no causal attribution, '
                       'capacity inference, or GPU performance claim.',
        candidate_share_denominator='All candidate selections, including selections '
                                    'with unchanged executed tokens; excludes base fallback.'
    ), indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
