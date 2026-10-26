"""Visualize the post-hoc SLO switch and its backlog sensitivity."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.www_capacity_pareto_20261015.run import OUT, RATES, read
from research.www_capacity_pareto_20261015.sensitivity import capacity, load_points

FIGURES = OUT / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib_cache'))

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

SPARE = '#BA3E3C'
PAST = '#3773A4'
S3 = '#16847C'
NATIVE = '#555D64'
OTHER = '#B5BEC4'
METHODS = {
    'arrival': ('SPARE', SPARE, 'o'),
    'past_future': ('Past-Future', PAST, 's'),
    's3_proxy': ('S3 proxy', S3, '^'),
    'native': ('vLLM V1', NATIVE, 'D'),
}

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'axes.linewidth': .7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'savefig.facecolor': 'white',
})


def scene_points(points, p95_limit, ref_rate):
    rows = []
    for (source, blocks, arm, budget), point in points.items():
        if (source, blocks) != ('conversation', 768):
            continue
        qps = capacity(point, p95_limit, 200., .05, 2.)
        rows.append(dict(setting='original' if p95_limit == 5000 else 'exploratory',
                         arm=arm, budget=budget, capacity_qps=qps,
                         reference_qps=ref_rate,
                         mean_ttft_ms=point['by_rate'][str(ref_rate)]['mean_ttft_ms'],
                         feasible=qps >= ref_rate))
    assert len(rows) == 48
    return rows


def pareto_axis(ax, rows, p95_limit):
    for row in rows:
        if row['arm'] in METHODS:
            continue
        ax.scatter(row['mean_ttft_ms'], row['capacity_qps'], s=15, c=OTHER,
                   alpha=.55, linewidths=0, zorder=1)
    for arm, (label, color, marker) in METHODS.items():
        group = [row for row in rows if row['arm'] == arm]
        ax.scatter([r['mean_ttft_ms'] for r in group],
                   [r['capacity_qps'] for r in group], s=20, c=color,
                   marker=marker, alpha=.38, linewidths=0, zorder=2)
        chosen = next(row for row in group if row['budget'] == 256)
        ax.scatter(chosen['mean_ttft_ms'], chosen['capacity_qps'], s=53,
                   c=color, marker=marker, edgecolors='white', linewidths=.7,
                   label=label, zorder=4)
    infeasible = [r for r in rows if not r['feasible']]
    if infeasible:
        ax.scatter([r['mean_ttft_ms'] for r in infeasible],
                   [r['capacity_qps'] for r in infeasible], s=25,
                   marker='x', c=NATIVE, linewidths=.7, zorder=5)
    ax.set_xlim((180, 1100) if p95_limit == 5000 else (200, 2600))
    ax.set_ylim(.25, 2.25)
    ax.set_yticks((.5, 1., 1.5, 2.))
    ax.grid(axis='y', color='#E7EBED', linewidth=.6)
    ax.set_axisbelow(True)
    ref = rows[0]['reference_qps']
    ax.set_xlabel(f'Mean TTFT at {ref:g} QPS (ms)')
    ax.set_ylabel('Serving capacity (QPS)')


def load_backlog():
    samples = [s for s in read(OUT / 'design.json')['samples']
               if s['split'] == 'test' and s['source'] == 'conversation']
    assert len(samples) == 50
    days = defaultdict(list)
    for index, sample in enumerate(samples):
        days[sample['day']].append(index)
    rates = (1.5, 2., 2.5)
    rows = []
    rng = np.random.default_rng(20261015)
    for arm in METHODS:
        if arm == 'native':
            continue
        traces = {rate: [] for rate in rates}
        for sample in samples:
            path = OUT / 'test' / f"{sample['name']}_c768_{arm}_t256.json"
            for row in read(path):
                if row['rate'] in traces:
                    traces[row['rate']].append(row['backlog_drift'])
        for rate in rates:
            values = np.asarray(traces[rate], dtype=float)
            assert len(values) == len(samples)
            boot = np.asarray([
                values[[int(index) for group in days.values()
                        for index in rng.choice(group, len(group))]].mean()
                for _ in range(1000)
            ])
            rows.append(dict(arm=arm, rate_qps=rate, mean_backlog=float(values.mean()),
                             ci_low=float(np.quantile(boot, .025)),
                             ci_high=float(np.quantile(boot, .975)),
                             limit=max(2., rate)))
    return rows


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    points = load_points('test')
    original = scene_points(points, 5000, .5)
    exploratory = scene_points(points, 12000, .75)
    backlog = load_backlog()
    write_csv(FIGURES / 'pareto_source_data.csv', original + exploratory)
    write_csv(FIGURES / 'backlog_source_data.csv', backlog)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.35))
    fig.subplots_adjust(left=.095, right=.975, bottom=.125, top=.83,
                        wspace=.30, hspace=.47)
    a, b, c, d = axes.flat
    pareto_axis(a, original, 5000)
    pareto_axis(b, exploratory, 12000)
    a.set_title('a   Original SLO: P95 TTFT <= 5 s', loc='left', fontweight='bold')
    b.set_title('b   Exploratory SLO: P95 TTFT <= 12 s', loc='left', fontweight='bold')

    thresholds = np.arange(1000., 15001., 250.)
    sweep_rows = []
    for arm in METHODS:
        if arm == 'native':
            continue
        point = points['conversation', 768, arm, 256]
        values = [capacity(point, limit, 200., .05, 2.) for limit in thresholds]
        name, color, marker = METHODS[arm]
        c.step(thresholds / 1000, values, where='post', label=name,
               color=color, linewidth=1.65)
        sweep_rows += [dict(arm=arm, p95_ttft_limit_ms=limit, capacity_qps=qps)
                       for limit, qps in zip(thresholds, values)]
    write_csv(FIGURES / 'capacity_sweep_source_data.csv', sweep_rows)
    c.axvspan(11.5, 14.75, color=SPARE, alpha=.08, linewidth=0)
    c.axvline(5, color='#7F888D', linestyle=':', linewidth=.9)
    c.axvline(12, color=SPARE, linestyle=':', linewidth=.9)
    c.text(5.15, 2.12, 'Original', color=NATIVE, fontsize=6)
    c.text(12.15, 2.12, 'Exploratory', color=SPARE, fontsize=6)
    c.set(xlim=(1, 15), ylim=(.2, 2.3),
          xlabel='P95 TTFT limit (s)', ylabel='Serving capacity (QPS)')
    c.set_yticks((.5, 1., 1.5, 2.))
    c.grid(axis='y', color='#E7EBED', linewidth=.6)
    c.set_axisbelow(True)
    c.set_title('c   Capacity changes at the chosen threshold',
                loc='left', fontweight='bold')

    for arm in METHODS:
        if arm == 'native':
            continue
        name, color, marker = METHODS[arm]
        group = [row for row in backlog if row['arm'] == arm]
        d.errorbar([row['rate_qps'] for row in group],
                   [row['mean_backlog'] for row in group],
                   yerr=[[row['mean_backlog'] - row['ci_low'] for row in group],
                         [row['ci_high'] - row['mean_backlog'] for row in group]],
                   marker=marker, markersize=4.5, linewidth=1.3,
                   capsize=2, color=color, label=name)
    d.plot([1.5, 2., 2.5], [2., 2., 2.5], color=NATIVE,
           linestyle='--', linewidth=.9, label='Backlog limit')
    d.set(xlim=(1.35, 2.65), xlabel='Offered load (QPS)',
          ylabel='Mean backlog growth (requests)')
    d.set_xticks((1.5, 2., 2.5))
    d.grid(axis='y', color='#E7EBED', linewidth=.6)
    d.set_axisbelow(True)
    d.set_title('d   Backlog gate at the switching load',
                loc='left', fontweight='bold')

    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center',
               bbox_to_anchor=(.5, .91), columnspacing=1.5,
               handletextpad=.4, fontsize=7)
    fig.suptitle('SLO tuning changes the Conversation / KV 768 comparison',
                 x=.095, y=.972, ha='left', fontsize=10, fontweight='bold')
    fig.text(.095, .935, 'All 48 budgets checked in each Pareto panel; '
             'colored large markers denote T256.', fontsize=6.5)
    fig.text(.095, .035,
             'Post-hoc sensitivity, CPU simulator. 50 test windows; '
             'error bars in d: day-stratified bootstrap 95% CI (1,000 draws).',
             fontsize=6.2, color=NATIVE)

    base = FIGURES / 'slo_switch_conversation_kv768'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=list(axes.flat), json_out=str(base) + '.alignment.json',
        overlay_svg=str(base) + '.alignment.svg', strict=True)
    for suffix, kwargs in (('.png', {'dpi': 350}),
                           ('.tiff', {'dpi': 600}),
                           ('.pdf', {}), ('.svg', {})):
        fig.savefig(str(base) + suffix, **kwargs)
    plt.close(fig)
    print(base)


if __name__ == '__main__':
    main()
