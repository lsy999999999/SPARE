"""Audit the fine-QPS pilot and render comparable capacity figures."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from research.www_kv_capacity_fine_20261016 import run
from research.www_kv_capacity_50wide_20261016.analyze import COLORS, LABELS, MARKERS

OUT = run.OUT
ANALYSIS = OUT / 'analysis'
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.labelsize': 9, 'axes.titlesize': 9,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})


def csv_write(name, rows):
    with (ANALYSIS / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def full_archive():
    rows = []
    for source in ('code', 'conversation'):
        for kv in run.CAPACITIES:
            for arm in run.METHODS:
                capacities = []
                for rate in run.old.RATES:
                    path = run.ARCHIVE / 'rates' / run.rate_path(source, kv, arm, rate).name
                    if not path.exists():
                        continue
                    raw = run.base.read(path)['rows']
                    valid = [r for r in raw if r['valid']]
                    pending = sum(r['pending_end'] for r in valid) / 50 if valid else None
                    drift = sum(r['backlog_drift'] for r in valid) / 50 if valid else None
                    passed = (len(valid) == 50 and
                              sum(r['p95_ttft_ms'] for r in valid) / 50 <= 12000 and
                              sum(r['p99_tbt_ms'] for r in valid) / 50 <= 200 and
                              ((pending < max(2, rate)) if run.PASS_KEY == 'passed_absolute'
                               else (drift <= max(2, rate))))
                    capacities.append((rate, passed))
                assert capacities and capacities[0][0] == .25
                first_failed = next((rate for rate, passed in capacities if not passed), None)
                capacity = max((rate for rate, passed in capacities
                                if passed and (first_failed is None or rate < first_failed)), default=0.)
                rows.append(dict(source=source, kv_blocks=kv, arm=arm,
                                 capacity_qps=capacity, next_failed_qps=first_failed,
                                 rate_files=len(capacities), windows=50,
                                 backlog_rule=('mean pending_end < max(2, rate)' if
                                               run.PASS_KEY == 'passed_absolute' else
                                               'mean backlog_drift <= max(2, rate)')))
    metric = 'absolute' if run.PASS_KEY == 'passed_absolute' else 'drift'
    csv_write(f'archived_50_window_{metric}_capacity.csv', rows)
    return rows


def load_fine():
    run.prepare()
    for shard in run.SHARDS:
        status = run.base.read(OUT / shard / 'status.json')
        assert status['stage'] == 'complete' and status['groups_complete'] == status['groups_total'] == 256
    by_source = []
    for source in ('code', 'conversation'):
        for kv in run.CAPACITIES:
            for arm in run.METHODS:
                group = run.base.read(OUT / 'groups' / f'{source}_c{kv}_{arm}.json')
                records = {}
                for path in (OUT / 'rates').glob(f'{source}_c{kv}_{arm}_r*.json'):
                    data = run.base.read(path)
                    rate = data['rows'][0]['rate']
                    assert rate not in records
                    assert data['summary'] == run.summary(data['rows'], rate)
                    records[rate] = data['summary'][run.PASS_KEY]
                first_failed = min((rate for rate, passed in records.items() if not passed),
                                   default=None)
                assert first_failed == group['next_failed_qps']
                expected_capacity = max((rate for rate, passed in records.items()
                                         if passed and rate < first_failed), default=0.)
                assert abs(expected_capacity - group['capacity_qps']) < 1e-9
                ref = run.base.read(run.rate_path(source, kv, arm, .5))
                assert ref['summary'] == run.summary(ref['rows'], .5)
                by_source.append(dict(source=source, kv_blocks=kv, arm=arm,
                                      method=LABELS[arm], capacity_qps=group['capacity_qps'],
                                      next_failed_qps=group['next_failed_qps'],
                                      resolution_qps=group['resolution_qps'],
                                      mean_ttft_ms=ref['summary']['mean_ttft_ms'],
                                      reference_feasible=group['reference_feasible'],
                                      nonmonotone_archived_rates=group['nonmonotone_archived_rates'],
                                      windows=2))
    csv_write('fine_by_source.csv', by_source)
    pairs = defaultdict(dict)
    for row in by_source:
        pairs[row['kv_blocks'], row['arm']][row['source']] = row
    macro = []
    for kv in run.CAPACITIES:
        for arm in run.METHODS:
            pair = pairs[kv, arm]
            macro.append(dict(kv_blocks=kv, arm=arm, method=LABELS[arm],
                              mean_ttft_ms=sum(pair[s]['mean_ttft_ms'] for s in pair)/2,
                              mean_capacity_qps=sum(pair[s]['capacity_qps'] for s in pair)/2,
                              reference_feasible=all(pair[s]['reference_feasible'] for s in pair),
                              windows_per_workload=2))
    csv_write('fine_macro.csv', macro)
    return by_source, macro


def legend():
    return [Line2D([], [], linestyle='None', marker=MARKERS[arm], markersize=6,
                   markerfacecolor='none' if arm == 'arrival' else COLORS[arm],
                   markeredgecolor=COLORS[arm], markeredgewidth=1.3,
                   label=LABELS[arm]) for arm in run.METHODS]


def plot_capacity(by_source):
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.1, right=.98, top=.94, bottom=.2, hspace=.15)
    for ax, source in zip(axes, ('code', 'conversation')):
        for arm in (*run.METHODS[1:], 'arrival'):
            rows = [r for r in by_source if r['source'] == source and r['arm'] == arm]
            ax.scatter([r['kv_blocks'] for r in rows], [r['capacity_qps'] for r in rows],
                       s=36 if arm == 'arrival' else 19, marker=MARKERS[arm],
                       facecolors='none' if arm == 'arrival' else COLORS[arm],
                       edgecolors=COLORS[arm], linewidths=1.2 if arm == 'arrival' else .45,
                       alpha=1 if arm == 'arrival' else .72,
                       zorder=4 if arm == 'arrival' else 2)
        ax.set_title(source.capitalize(), loc='left')
        ax.set_ylabel('Capacity (QPS)')
        ax.grid(axis='y', color='#e5e7e9', linewidth=.45)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel('KV capacity (blocks)')
    axes[-1].set_xlim(500, 8350)
    axes[-1].set_xticks((768, 1024, 2048, 3072, 4096, 5632, 6656, 8192))
    if run.PASS_KEY == 'passed_drift':
        axes[0].set_ylim(0, 6.4)
        axes[0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(1.))
    else:
        axes[0].set_ylim(0, 2.15)
        axes[0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(.2))
    fig.legend(handles=legend(), ncol=4, loc='lower center',
               bbox_to_anchor=(.53, .02), fontsize=7, columnspacing=1.25)
    fig.savefig(ANALYSIS / 'fine_capacity_by_kv.png', dpi=300)
    fig.savefig(ANALYSIS / 'fine_capacity_by_kv.pdf')
    plt.close(fig)


def plot_fixed_kv(macro):
    chosen = (768, 1024, 4096, 8192)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.1, right=.98, top=.93, bottom=.2, wspace=.13, hspace=.2)
    for ax, kv in zip(axes.flat, chosen):
        for row in (r for r in macro if r['kv_blocks'] == kv):
            arm = row['arm']
            ax.scatter(row['mean_ttft_ms'], row['mean_capacity_qps'],
                       marker=MARKERS[arm], s=70 if arm == 'arrival' else 38,
                       facecolors='none' if arm == 'arrival' else COLORS[arm],
                       edgecolors=COLORS[arm], linewidths=1.4 if arm == 'arrival' else .55,
                       alpha=1 if arm == 'arrival' else .8,
                       zorder=4 if arm == 'arrival' else 2)
        ax.set_title(f'KV = {kv} blocks', loc='left')
        ax.set_xscale('log')
        ax.grid(axis='y', color='#e5e7e9', linewidth=.45)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel('Mean capacity (QPS)')
    for ax in axes[-1]:
        ax.set_xlabel('Mean TTFT at 0.5 QPS (ms)')
    fig.legend(handles=legend(), ncol=4, loc='lower center',
               bbox_to_anchor=(.53, .02), fontsize=7, columnspacing=1.25)
    fig.savefig(ANALYSIS / 'fine_fixed_kv_pareto.png', dpi=300)
    fig.savefig(ANALYSIS / 'fine_fixed_kv_pareto.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rule', choices=('absolute', 'drift'), default='absolute')
    args = parser.parse_args()
    run.configure(args.rule)
    OUT = run.OUT
    ANALYSIS = OUT / 'analysis'
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    archive = full_archive()
    by_source, macro = load_fine()
    plot_capacity(by_source)
    plot_fixed_kv(macro)
    print(dict(archived_rows=len(archive), fine_rows=len(by_source), macro_rows=len(macro),
               nonmonotone=sum(r['nonmonotone_archived_rates'] for r in by_source),
               resolution_le_0_1=sum(r['resolution_qps'] is not None and
                                     r['resolution_qps'] <= .10001 for r in by_source)))
