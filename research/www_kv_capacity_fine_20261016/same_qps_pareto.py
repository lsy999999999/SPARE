"""Audit same-KV Pareto status with TTFT measured at a shared 0.5 QPS."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from research.www_kv_capacity_fine_20261016 import analyze, run
from research.www_kv_capacity_50wide_20261016.analyze import COLORS, MARKERS

OUT = run.ROOT / 'artifacts/www_kv_capacity_fine_drift_20261016_r1/analysis'


def read_csv(path):
    return list(csv.DictReader(path.open(newline='')))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coarse_two_window():
    source_rows = read_csv(OUT / 'fine_by_source.csv')
    paired = defaultdict(dict)
    for item in source_rows:
        source, kv, arm = item['source'], int(item['kv_blocks']), item['arm']
        known = []
        for rate in run.old.RATES:
            path = run.rate_path(source, kv, arm, rate)
            if path.exists():
                data = run.base.read(path)
                assert len(data['rows']) == 2
                assert data['summary'] == run.summary(data['rows'], rate)
                known.append((rate, data['summary']['passed_drift']))
        assert known and known[0][0] == .25
        first_failed = next((rate for rate, passed in known if not passed), None)
        assert first_failed is not None
        capacity = max((rate for rate, passed in known
                        if passed and rate < first_failed), default=0.)
        paired[kv, arm][source] = dict(capacity_qps=capacity,
                                       mean_ttft_ms=float(item['mean_ttft_ms']))
    output = []
    for kv in run.CAPACITIES:
        for arm in run.METHODS:
            rows = paired[kv, arm]
            output.append(dict(kv_blocks=kv, arm=arm,
                               mean_ttft_ms=sum(x['mean_ttft_ms'] for x in rows.values())/2,
                               mean_capacity_qps=sum(x['capacity_qps'] for x in rows.values())/2,
                               windows_per_workload=2, reference_qps=.5))
    write_csv(OUT / 'same_qps_coarse_grid_two_window_macro.csv', output)
    return output


def dominates(first, second):
    a_ttft = float(first['mean_ttft_ms'])
    b_ttft = float(second['mean_ttft_ms'])
    a_capacity = float(first['mean_capacity_qps'])
    b_capacity = float(second['mean_capacity_qps'])
    return (a_ttft <= b_ttft + 1e-8 and a_capacity >= b_capacity - 1e-9
            and (a_ttft < b_ttft - 1e-8 or a_capacity > b_capacity + 1e-9))


def status(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[int(row['kv_blocks'])][row['arm']] = row
    result = {}
    for kv in run.CAPACITIES:
        group = groups[kv]
        assert set(group) == set(run.METHODS)
        spare = group['arrival']
        result[kv] = dict(
            spare_capacity_qps=float(spare['mean_capacity_qps']),
            spare_ttft_ms=float(spare['mean_ttft_ms']),
            dominated_by='; '.join(group[arm].get('method', arm) for arm in run.METHODS[1:]
                                   if dominates(group[arm], spare)),
            best_baseline_capacity_qps=max(float(group[arm]['mean_capacity_qps'])
                                           for arm in run.METHODS[1:]))
    return result


def plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.92, bottom=.26, wspace=.16)
    for ax in axes:
        for arm in (*run.METHODS[1:], 'arrival'):
            group = [r for r in rows if r['arm'] == arm]
            ax.scatter([float(r['mean_ttft_ms']) for r in group],
                       [float(r['mean_capacity_qps']) for r in group],
                       marker=MARKERS[arm], s=58 if arm == 'arrival' else 22,
                       facecolors='none' if arm == 'arrival' else COLORS[arm],
                       edgecolors=COLORS[arm], linewidths=1.4 if arm == 'arrival' else .5,
                       alpha=1 if arm == 'arrival' else .7,
                       zorder=4 if arm == 'arrival' else 2)
        ax.set_xlabel('Mean TTFT at shared 0.5 QPS (ms)')
        ax.grid(axis='y', color='#e5e7e9', linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_title('All methods', loc='left')
    axes[0].set_xscale('log')
    axes[0].set_ylabel('Mean capacity (QPS)')
    axes[1].set_title('Low-TTFT detail', loc='left')
    axes[1].set_xlim(280, 520)
    axes[1].set_xticks((300, 350, 400, 450, 500))
    axes[0].set_ylim(.4, 5.25)
    fig.legend(handles=analyze.legend(), ncol=4, loc='lower center',
               bbox_to_anchor=(.53, .03), fontsize=7, columnspacing=1.25)
    fig.savefig(OUT / 'same_qps_fine_all_kv.png', dpi=300)
    fig.savefig(OUT / 'same_qps_fine_all_kv.pdf')
    plt.close(fig)


def main():
    run.configure('drift')
    run.prepare()
    fine = read_csv(OUT / 'fine_macro.csv')
    coarse = coarse_two_window()
    archived = read_csv(run.ARCHIVE / 'analysis/macro_averages.csv')
    statuses = {name: status(rows) for name, rows in
                (('fine_2_windows', fine), ('coarse_2_windows', coarse),
                 ('coarse_50_windows', archived))}
    output = []
    for kv in run.CAPACITIES:
        row = dict(kv_blocks=kv)
        for name, results in statuses.items():
            item = results[kv]
            row[f'{name}_spare_capacity_qps'] = item['spare_capacity_qps']
            row[f'{name}_spare_ttft_ms'] = item['spare_ttft_ms']
            row[f'{name}_best_baseline_capacity_qps'] = item['best_baseline_capacity_qps']
            row[f'{name}_dominated_by'] = item['dominated_by']
            row[f'{name}_spare_nondominated'] = not bool(item['dominated_by'])
        output.append(row)
    write_csv(OUT / 'same_qps_pareto_status_by_kv.csv', output)
    plot(fine)
    summary = {name: dict(nondominated=sum(not v['dominated_by'] for v in values.values()),
                          dominated_kv=[kv for kv, v in values.items() if v['dominated_by']])
               for name, values in statuses.items()}
    (OUT / 'same_qps_pareto_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(summary)


if __name__ == '__main__':
    main()
