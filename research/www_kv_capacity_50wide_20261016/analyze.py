"""Aggregate completed 50-window KV scans and draw the average objectives."""
from __future__ import annotations

import csv
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/www_kv_capacity_50wide_20261016_r1'
ANALYSIS = OUT / 'analysis'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'mpl_cache'))

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

from research.www_kv_capacity_50wide_20261016 import run
from research.www_kv_extreme_stress_20261016.analyze import STYLE

LABELS = {arm: label for arm, label, *_ in STYLE}
COLORS = {
    'arrival': '#bd3638', 'native': '#606b73', 'sarathi': '#8da8b4',
    'sola_rule': '#bc8651', 'uniboost_static': '#8d78a0',
    'past_future': '#218983', 's3_proxy': '#c37757',
    'apt_serve_kv_rule': '#a19b5d',
}
MARKERS = dict(zip(run.METHODS, ('o', 's', '^', 'v', 'D', 'P', 'X', '*')))


def configure(profile):
    global OUT, ANALYSIS
    run.configure(profile)
    OUT = run.OUT
    ANALYSIS = OUT / 'analysis'
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'legend.frameon': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
})


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def load():
    design = run.prepare()
    for shard in run.SHARDS:
        status = run.base.read(OUT / shard / 'status.json')
        assert status['stage'] == 'complete' and not status['errors']
        assert status['groups_complete'] == status['groups_total'] == 2 * len(run.SHARDS[shard]) * len(run.METHODS)
    assert len(list((OUT / 'groups').glob('*.json'))) == 2 * len(run.CAPACITIES) * len(run.METHODS)

    by_source = []
    for kv in run.CAPACITIES:
        for arm in run.METHODS:
            for source in ('code', 'conversation'):
                group = run.base.read(OUT / 'groups' / f'{source}_c{kv}_{arm}.json')
                reference = run.base.read(OUT / 'rates' / f'{source}_c{kv}_{arm}_r0.5.json')
                assert len(reference['rows']) == 50
                assert len({r['name'] for r in reference['rows']}) == 50
                assert reference['summary'] == run.summary(reference['rows'], .5)
                assert group['reference_feasible'] == (group['capacity_qps'] >= .5)
                by_source.append(dict(
                    source=source, kv_blocks=kv, arm=arm, method=LABELS[arm],
                    capacity_qps=group['capacity_qps'],
                    next_failed_qps=group['next_failed_qps'],
                    capacity_is_lower_bound=group['capacity_is_lower_bound'],
                    mean_ttft_ms=reference['summary']['mean_ttft_ms'],
                    reference_feasible=group['reference_feasible'],
                    windows=50,
                ))

    paired = defaultdict(dict)
    for row in by_source:
        paired[row['kv_blocks'], row['arm']][row['source']] = row
    means = []
    for kv in run.CAPACITIES:
        for arm in run.METHODS:
            pair = paired[kv, arm]
            assert set(pair) == {'code', 'conversation'}
            values = [pair[source] for source in ('code', 'conversation')]
            assert all(v['reference_feasible'] for v in values)
            means.append(dict(
                kv_blocks=kv, arm=arm, method=LABELS[arm],
                mean_ttft_ms=sum(v['mean_ttft_ms'] for v in values) / 2,
                mean_capacity_qps=sum(v['capacity_qps'] for v in values) / 2,
                reference_qps=.5, windows_per_workload=50,
                reference_feasible=True,
            ))
    assert len(means) == len(run.CAPACITIES) * len(run.METHODS)
    return by_source, means


def dominates(first, second):
    return (first['mean_ttft_ms'] <= second['mean_ttft_ms'] + 1e-8
            and first['mean_capacity_qps'] >= second['mean_capacity_qps']
            and (first['mean_ttft_ms'] < second['mean_ttft_ms'] - 1e-8
                 or first['mean_capacity_qps'] > second['mean_capacity_qps']))


def comparisons(means):
    rows = []
    for kv in run.CAPACITIES:
        ours = next(p for p in means if p['kv_blocks'] == kv and p['arm'] == 'arrival')
        rivals = [p for p in means if p['kv_blocks'] == kv and p['arm'] != 'arrival']
        rows.append(dict(
            kv_blocks=kv, spare_mean_ttft_ms=ours['mean_ttft_ms'],
            spare_mean_capacity_qps=ours['mean_capacity_qps'],
            best_baseline_capacity_qps=max(p['mean_capacity_qps'] for p in rivals),
            dominating_methods='; '.join(p['method'] for p in rivals if dominates(p, ours)),
            tied_methods='; '.join(p['method'] for p in rivals
                                   if p['mean_capacity_qps'] == ours['mean_capacity_qps']
                                   and abs(p['mean_ttft_ms'] - ours['mean_ttft_ms']) < 1e-8),
        ))
    return rows


def paired_ratios(means):
    rows = []
    for kv in run.CAPACITIES:
        ours = next(p for p in means if p['kv_blocks'] == kv and p['arm'] == 'arrival')
        for baseline in (p for p in means if p['kv_blocks'] == kv and p['arm'] != 'arrival'):
            rows.append(dict(
                kv=kv, method=baseline['method'], arm=baseline['arm'],
                spare_ttft_ms=ours['mean_ttft_ms'],
                spare_capacity_qps=ours['mean_capacity_qps'],
                baseline_ttft_ms=baseline['mean_ttft_ms'],
                baseline_capacity_qps=baseline['mean_capacity_qps'],
                ttft_ratio_baseline_over_spare=baseline['mean_ttft_ms'] / ours['mean_ttft_ms'],
                capacity_ratio_spare_over_baseline=(ours['mean_capacity_qps'] /
                                                   baseline['mean_capacity_qps']),
                baseline_dominates_spare=dominates(baseline, ours),
            ))
    return rows


def legend_handles():
    return [Line2D([], [], linestyle='None', marker=MARKERS[arm], markersize=7 if arm == 'arrival' else 5,
                   markerfacecolor='none' if arm == 'arrival' else COLORS[arm],
                   markeredgecolor=COLORS[arm], markeredgewidth=1.5 if arm == 'arrival' else .5,
                   label=LABELS[arm]) for arm in run.METHODS]


def plot_scatter(means):
    stem = ANALYSIS / 'mean_ttft_capacity_scatter'
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.subplots_adjust(left=.11, right=.97, top=.94, bottom=.28)
    for arm in (*run.METHODS[1:], 'arrival'):
        series = [p for p in means if p['arm'] == arm]
        is_spare = arm == 'arrival'
        ax.scatter([p['mean_ttft_ms'] for p in series],
                   [p['mean_capacity_qps'] for p in series],
                   marker=MARKERS[arm], s=75 if is_spare else 27,
                   facecolor='none' if is_spare else COLORS[arm],
                   edgecolor=COLORS[arm] if is_spare else 'white',
                   linewidth=1.5 if is_spare else .3,
                   alpha=1 if is_spare else .62, zorder=5 if is_spare else 2)
    annotations = {
        1024: ('1024', (9, -10)), 1280: ('1280', (9, 10)),
        2048: ('1536, 2048', (9, 8)),
        3072: ('3072', (9, 9)),
        8192: ('4096, 8192', (9, 9)),
        6144: ('6144', (9, 9)),
    }
    if len(run.CAPACITIES) > 21:
        annotations = {
            672: ('672', (9, 8)),
            768: ('768', (9, -12)),
            896: ('896', (60, 8)),
            1024: ('1024', (70, 0)),
            1280: ('1280', (9, 9)),
            1536: ('1536-2048', (80, 8)),
            2304: ('2304-3584', (9, 9)),
            3840: ('3840-5376; 6912-8192', (9, 9)),
            5632: ('5632-6656', (9, 9)),
        }
    for kv, (label, offset) in annotations.items():
        point = next(p for p in means if p['kv_blocks'] == kv and p['arm'] == 'arrival')
        leader = (dict(arrowstyle='-', color=COLORS['arrival'], lw=.55,
                       shrinkA=3, shrinkB=5)
                  if len(run.CAPACITIES) > 21 and kv in (896, 1024, 1536)
                  else None)
        ax.annotate(label, (point['mean_ttft_ms'], point['mean_capacity_qps']),
                    xytext=offset, textcoords='offset points', color=COLORS['arrival'],
                    fontsize=7, va='center', zorder=6, arrowprops=leader)
    ax.set_xscale('log')
    ax.set_xticks((300, 500, 1000, 2000, 3000), labels=('300', '500', '1000', '2000', '3000'))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlim(280, 3600)
    ax.set_ylim(0, 5)
    ax.set_xlabel('Mean TTFT at 0.5 QPS (ms; log scale)')
    ax.set_ylabel('Mean serving capacity (QPS)')
    ax.grid(axis='y', color='#e4e7e9', linewidth=.5)
    ax.set_axisbelow(True)
    fig.legend(handles=legend_handles(), ncol=4, loc='lower center',
               bbox_to_anchor=(.52, .05), fontsize=7, columnspacing=1.3)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=[ax],
                                       json_out=str(stem) + '.alignment.json', strict=True)
    fig.savefig(str(stem) + '.png', dpi=350)
    fig.savefig(str(stem) + '.pdf')
    fig.savefig(str(stem) + '.svg')
    plt.close(fig)


def plot_by_kv(means):
    stem = ANALYSIS / 'mean_metrics_by_kv'
    fig, axes = plt.subplots(2, 1,
                             figsize=(14 if len(run.CAPACITIES) > 21 else 10, 6.9),
                             sharex=True)
    fig.subplots_adjust(left=.095, right=.985, top=.96, bottom=.23, hspace=.22)
    positions = {kv: i for i, kv in enumerate(run.CAPACITIES)}
    for arm in (*run.METHODS[1:], 'arrival'):
        series = [p for p in means if p['arm'] == arm]
        x = [positions[p['kv_blocks']] for p in series]
        for ax, field in zip(axes, ('mean_ttft_ms', 'mean_capacity_qps')):
            ax.plot(x, [p[field] for p in series], color=COLORS[arm],
                    marker=MARKERS[arm], markersize=5 if arm == 'arrival' else 3,
                    markerfacecolor='white' if arm == 'arrival' else COLORS[arm],
                    markeredgewidth=1.3 if arm == 'arrival' else .5,
                    linewidth=1.8 if arm == 'arrival' else 1,
                    alpha=1 if arm == 'arrival' else .7,
                    zorder=5 if arm == 'arrival' else 2)
    axes[0].set_yscale('log')
    axes[0].tick_params(axis='y', labelsize=8)
    axes[0].set_ylim(275, 4100)
    axes[0].set_ylabel('Mean TTFT at 0.5 QPS (ms; log scale)')
    axes[1].set_ylim(0, 4.9)
    axes[1].set_ylabel('Mean serving capacity (QPS)')
    axes[1].set_xlim(-.3, len(run.CAPACITIES)-.7)
    axes[1].set_xticks(range(len(run.CAPACITIES)),
                       labels=[str(kv) for kv in run.CAPACITIES], rotation=55, ha='right',
                       rotation_mode='anchor')
    axes[1].set_xlabel('KV capacity (blocks; discrete scan levels)')
    for ax in axes:
        ax.grid(axis='y', color='#e4e7e9', linewidth=.5)
        ax.set_axisbelow(True)
    fig.legend(handles=legend_handles(), ncol=4, loc='lower center',
               bbox_to_anchor=(.52, .04), fontsize=7, columnspacing=1.7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=axes,
                                       json_out=str(stem) + '.alignment.json', strict=True)
    fig.savefig(str(stem) + '.png', dpi=350)
    fig.savefig(str(stem) + '.pdf')
    fig.savefig(str(stem) + '.svg')
    plt.close(fig)


def plot_fixed_kv_pareto(means):
    stem = ANALYSIS / 'fixed_kv_pareto'
    selected = (768, 1024, 5632, 8192)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6))
    fig.subplots_adjust(left=.085, right=.975, top=.93, bottom=.19,
                        hspace=.38, wspace=.22)
    for ax, kv in zip(axes.flat, selected):
        points = [p for p in means if p['kv_blocks'] == kv]
        frontier = sorted({(p['mean_ttft_ms'], p['mean_capacity_qps'])
                           for p in points if not any(dominates(q, p) for q in points if q is not p)})
        if len(frontier) > 1:
            ax.plot([x for x, _ in frontier], [y for _, y in frontier],
                    color='#343a40', linestyle='--', linewidth=1, zorder=1)
        for arm in (*run.METHODS[1:], 'arrival'):
            p = next(p for p in points if p['arm'] == arm)
            is_spare = arm == 'arrival'
            ax.scatter(p['mean_ttft_ms'], p['mean_capacity_qps'],
                       marker=MARKERS[arm], s=76 if is_spare else 42,
                       facecolor='white' if is_spare else COLORS[arm],
                       edgecolor=COLORS[arm] if is_spare else 'white',
                       linewidth=1.6 if is_spare else .4,
                       zorder=5 if is_spare else 3)
        xs = [p['mean_ttft_ms'] for p in points]
        if max(xs) / min(xs) > 1.8:
            ax.set_xscale('log')
            ticks = [t for t in (300, 400, 500, 600, 800, 1000)
                     if min(xs) * .94 < t < max(xs) * 1.06]
            ax.set_xticks(ticks, labels=[str(t) for t in ticks])
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlim(min(xs) * .94, max(xs) * 1.06)
        ax.set_ylim(0, max(p['mean_capacity_qps'] for p in points) + .45)
        ax.set_title(f'KV = {kv} blocks')
        ax.set_xlabel('Mean TTFT at 0.5 QPS (ms)')
        ax.set_ylabel('Mean capacity (QPS)')
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(axis='y', color='#e4e7e9', linewidth=.5)
        ax.set_axisbelow(True)
    fig.legend(handles=legend_handles(), ncol=4, loc='lower center',
               bbox_to_anchor=(.51, .035), fontsize=7, columnspacing=1.3)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=list(axes.flat),
                                       json_out=str(stem) + '.alignment.json', strict=True)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(str(stem) + '.' + suffix, dpi=350 if suffix == 'png' else None)
    plt.close(fig)


def plot_dominance_pairs(rows):
    stem = ANALYSIS / 'same_kv_dominance_ratios'
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    fig.subplots_adjust(left=.12, right=.97, top=.95, bottom=.28)
    xkey = 'ttft_ratio_baseline_over_spare'
    ykey = 'capacity_ratio_spare_over_baseline'
    for arm in run.METHODS[1:]:
        for low in (False, True):
            series = [r for r in rows if r['arm'] == arm and (r['kv'] < 1024) == low]
            ax.scatter([r[xkey] for r in series], [r[ykey] for r in series],
                       color=COLORS[arm], marker='x' if low else MARKERS[arm],
                       s=27 if low else 35, alpha=.75, linewidth=.8, zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(.65, max(r[xkey] for r in rows) * 1.08)
    ax.set_ylim(.65, max(r[ykey] for r in rows) * 1.1)
    for axis, candidates in ((ax.xaxis, (.75, 1, 1.5, 2, 3, 4)),
                             (ax.yaxis, (.75, 1, 1.5, 2, 3, 4))):
        low, high = (ax.get_xlim() if axis is ax.xaxis else ax.get_ylim())
        ticks = [t for t in candidates if low < t < high]
        axis.set_ticks(ticks, labels=[f'{t:g}' for t in ticks])
        axis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.axvline(1, color='#535b60', linestyle='--', linewidth=.8)
    ax.axhline(1, color='#535b60', linestyle='--', linewidth=.8)
    ax.set_xlabel('Baseline mean TTFT / SPARE mean TTFT')
    ax.set_ylabel('SPARE capacity / baseline capacity')
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(color='#e4e7e9', linewidth=.5)
    ax.set_axisbelow(True)
    handles = legend_handles()[1:]
    handles.append(Line2D([], [], linestyle='None', marker='x', color='#555555',
                          label='KV < 1024'))
    fig.legend(handles=handles, ncol=4, loc='lower center',
               bbox_to_anchor=(.52, .055), fontsize=7, columnspacing=1.2)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=[ax],
                                       json_out=str(stem) + '.alignment.json', strict=True)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(str(stem) + '.' + suffix, dpi=350 if suffix == 'png' else None)
    plt.close(fig)


def report(rows):
    lines = [
        f'# {len(run.CAPACITIES)}-KV scan, 50 windows per workload', '',
        'Fixed SPARE and seven adapter/rule comparisons. Code and Conversation are averaged equally.',
        'Each method-KV point uses 50 paired test windows per workload. Mean TTFT is measured at',
        '0.5 QPS; serving capacity is the largest passing arrival-rate grid point under P95 TTFT',
        '<=12 s, P99 TBT <=200 ms, and the existing backlog criterion. Capacity is a macro-average',
        'of two workloads, not the capacity of a mixed request stream. The SLO is exploratory.',
        'Every plotted point has an exact KV tick in mean_metrics_by_kv; the scatter intentionally',
        'keeps coincident coordinates overlapped rather than jittering them.',
        'The 672, 768, and 896-block points are low-capacity context. The largest individual',
        'request needs 499 blocks, so 1024 blocks is the first scan point that can hold any',
        'two largest requests at once.', '',
        '|KV blocks|SPARE mean TTFT (ms)|SPARE mean capacity (QPS)|Best baseline capacity (QPS)|Dominates SPARE|Ties SPARE|',
        '|---:|---:|---:|---:|---|---|',
    ]
    for row in rows:
        lines.append(f"|{row['kv_blocks']}|{row['spare_mean_ttft_ms']:.2f}|"
                     f"{row['spare_mean_capacity_qps']:g}|{row['best_baseline_capacity_qps']:g}|"
                     f"{row['dominating_methods'] or '-'}|{row['tied_methods'] or '-'}|")
    lines += ['', f"SPARE non-dominated at {sum(not r['dominating_methods'] for r in rows)}/{len(rows)}"
              ' KV levels among these eight fixed implementations. The 4.5 QPS maximum occurs',
              'at 5632-6656 blocks, then falls to 4.0 QPS at 6912-8192 blocks. At 5 QPS,',
              'Conversation backlog drift moves from 4.98 at 6656 to 5.08 at 6912 against',
              'the 5.0 limit; the measured drop is close to this exploratory threshold.',
              'UniBoost is a fixed-gamma rule adaptation, and S3 is a packing proxy; neither',
              'is a full reproduction of its original system. Point estimates only; no new',
              'confidence intervals were calculated. The 50-window six-KV historical',
              'comparison includes four additional implementations and is a distinct set.']
    (ANALYSIS / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    by_source, means = load()
    comparisons_rows = comparisons(means)
    write_csv(ANALYSIS / 'by_source.csv', by_source)
    write_csv(ANALYSIS / 'macro_averages.csv', means)
    write_csv(ANALYSIS / 'plot_data.csv', [dict(
        kv=row['kv_blocks'], method=row['method'],
        ttft_ms=row['mean_ttft_ms'], capacity_qps=row['mean_capacity_qps'],
    ) for row in means])
    write_csv(ANALYSIS / 'same_kv_comparisons.csv', comparisons_rows)
    if len(run.CAPACITIES) > 21:
        pairs = paired_ratios(means)
        write_csv(ANALYSIS / 'same_kv_dominance_ratios.csv', pairs)
        plot_fixed_kv_pareto(means)
        plot_dominance_pairs(pairs)
    plot_scatter(means)
    plot_by_kv(means)
    report(comparisons_rows)
    print(f'{len(means)} method-KV points; '
          f'{sum(not r["dominating_methods"] for r in comparisons_rows)}/'
          f'{len(run.CAPACITIES)} SPARE points non-dominated')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('wide', 'dense'), default='wide')
    args = parser.parse_args()
    configure(args.profile)
    main()
