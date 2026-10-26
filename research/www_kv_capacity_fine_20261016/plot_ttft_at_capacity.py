"""Plot TTFT measured at each method's own passing capacity QPS."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from research.www_kv_capacity_fine_20261016 import analyze, run
from research.www_kv_capacity_50wide_20261016.analyze import COLORS, LABELS, MARKERS


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def per_source(archive, windows):
    output = []
    for record in archive:
        source, kv, arm = record['source'], int(record['kv_blocks']), record['arm']
        capacity = float(record['capacity_qps'])
        if capacity:
            filename = run.rate_path(source, kv, arm, capacity).name
            path = ((run.OUT if windows == 2 else run.ARCHIVE) / 'rates' / filename)
            rate_data = run.base.read(path)
            assert len(rate_data['rows']) == windows
            assert all(r['rate'] == capacity and r['valid'] for r in rate_data['rows'])
            ttft = sum(r['mean_ttft_ms'] for r in rate_data['rows']) / windows
        else:
            ttft = None
        output.append(dict(source=source, kv_blocks=kv, arm=arm,
                           method=LABELS[arm], capacity_qps=capacity,
                           mean_ttft_at_capacity_ms=ttft,
                           measured_qps=capacity if capacity else None,
                           ttft_available=ttft is not None, windows=windows))
    return output


def macro(by_source, windows):
    pairs = defaultdict(dict)
    for row in by_source:
        pairs[row['kv_blocks'], row['arm']][row['source']] = row
    output = []
    for kv in run.CAPACITIES:
        for arm in run.METHODS:
            code = pairs[kv, arm]['code']
            conversation = pairs[kv, arm]['conversation']
            usable = code['ttft_available'] and conversation['ttft_available']
            output.append(dict(kv_blocks=kv, arm=arm, method=LABELS[arm],
                               code_capacity_qps=code['capacity_qps'],
                               conversation_capacity_qps=conversation['capacity_qps'],
                               mean_capacity_qps=(code['capacity_qps'] + conversation['capacity_qps'])/2,
                               mean_ttft_at_capacity_ms=(
                                   (code['mean_ttft_at_capacity_ms'] +
                                    conversation['mean_ttft_at_capacity_ms'])/2 if usable else None),
                               ttft_available=usable, windows_per_workload=windows))
    return output


def points(ax, rows):
    for arm in (*run.METHODS[1:], 'arrival'):
        selected = [r for r in rows if r['arm'] == arm and r['ttft_available']]
        ax.scatter([r['mean_ttft_at_capacity_ms'] for r in selected],
                   [r['mean_capacity_qps'] for r in selected],
                   marker=MARKERS[arm], s=72 if arm == 'arrival' else 35,
                   facecolors='none' if arm == 'arrival' else COLORS[arm],
                   edgecolors=COLORS[arm], linewidths=1.5 if arm == 'arrival' else .55,
                   alpha=1 if arm == 'arrival' else .75,
                   zorder=4 if arm == 'arrival' else 2)


def fixed_kv(rows, windows, target):
    chosen = (768, 1024, 4096, 8192)
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.1, right=.98, top=.94, bottom=.21, wspace=.15, hspace=.19)
    for ax, kv in zip(axes.flat, chosen):
        points(ax, [r for r in rows if r['kv_blocks'] == kv])
        ax.set_title(f'KV = {kv} blocks', loc='left')
        ax.set_xscale('log')
        ax.grid(axis='y', color='#e5e7e9', linewidth=.5)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel('Mean capacity (QPS)')
    for ax in axes[-1]:
        ax.set_xlabel('Mean TTFT at capacity QPS (ms)')
    fig.legend(handles=analyze.legend(), ncol=4, loc='lower center',
               bbox_to_anchor=(.53, .025), fontsize=7, columnspacing=1.25)
    fig.savefig(target.with_suffix('.png'), dpi=300)
    fig.savefig(target.with_suffix('.pdf'))
    plt.close(fig)


def all_kv(rows, target):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    fig.subplots_adjust(left=.11, right=.98, top=.95, bottom=.25)
    points(ax, rows)
    ax.set_xscale('log')
    ax.set_xlabel('Mean TTFT at capacity QPS (ms)')
    ax.set_ylabel('Mean capacity (QPS)')
    ax.grid(axis='y', color='#e5e7e9', linewidth=.5)
    ax.set_axisbelow(True)
    fig.legend(handles=analyze.legend(), ncol=4, loc='lower center',
               bbox_to_anchor=(.53, .035), fontsize=7, columnspacing=1.25)
    fig.savefig(target.with_suffix('.png'), dpi=300)
    fig.savefig(target.with_suffix('.pdf'))
    plt.close(fig)


def main(rule):
    run.configure(rule)
    run.prepare()
    analyze.OUT = run.OUT
    analyze.ANALYSIS = run.OUT / 'analysis'
    analysis = analyze.ANALYSIS
    analysis.mkdir(parents=True, exist_ok=True)
    old = analyze.full_archive()
    fine, _ = analyze.load_fine()
    for records, windows, suffix in ((fine, 2, 'fine'), (old, 50, 'archived_50_window')):
        sources = per_source(records, windows)
        means = macro(sources, windows)
        assert len(sources) == 512 and len(means) == 256
        missing = sum(not r['ttft_available'] for r in sources)
        assert missing == sum(r['capacity_qps'] == 0 for r in sources)
        write_csv(analysis / f'ttft_at_capacity_{suffix}_by_source.csv', sources)
        write_csv(analysis / f'ttft_at_capacity_{suffix}_macro.csv', means)
        fixed_kv(means, windows, analysis / f'ttft_at_capacity_{suffix}_fixed_kv')
        all_kv(means, analysis / f'ttft_at_capacity_{suffix}_all_kv')
        print(f'{rule}, {windows} windows: {missing} source points have no passing QPS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rule', choices=('absolute', 'drift'), default='absolute')
    args = parser.parse_args()
    main(args.rule)
