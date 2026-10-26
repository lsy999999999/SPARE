"""Audit the 50-window fine scan and plot same-KV Pareto comparisons."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_kv_fine50_mpl')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

from research.www_kv_capacity_fine50_20261016 import run
from research.www_kv_capacity_50wide_20261016.analyze import COLORS, LABELS, MARKERS
from research.www_kv_capacity_fine_20261016.same_qps_pareto import dominates

OUT = run.OUT
ANALYSIS = OUT / 'analysis'
SOURCES = ('code', 'conversation')

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
})


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load():
    design = run.prepare()
    assert tuple(design['capacities']) == run.CAPACITIES
    assert len(design['samples']) == 100
    for shard in run.SHARDS:
        status = run.base.read(OUT / shard / 'status.json')
        assert status['stage'] == 'complete' and not status['errors']
        assert status['groups_complete'] == status['groups_total'] == 2 * len(run.SHARDS[shard]) * 8
    assert len(list((OUT / 'groups').glob('*.json'))) == 512

    by_source = []
    for kv in run.CAPACITIES:
        for arm in run.METHODS:
            for source in SOURCES:
                group = run.base.read(OUT / 'groups' / f'{source}_c{kv}_{arm}.json')
                assert group['resolution_qps'] <= .020001
                assert group['windows_per_workload'] == 50
                for rate_str, check in group['rates_tested'].items():
                    rate = float(rate_str)
                    record = run.base.read(run.rate_path(source, kv, arm, rate))
                    assert len(record['rows']) == len({r['name'] for r in record['rows']}) == 50
                    assert record['summary'] == run.old.summary(record['rows'], rate)
                    assert record['summary']['passed'] == check['passed']
                reference = run.base.read(run.ARCHIVE / 'rates' /
                                          run.rate_path(source, kv, arm, .5).name)
                assert len(reference['rows']) == 50
                assert reference['summary'] == run.old.summary(reference['rows'], .5)
                by_source.append(dict(
                    source=source, kv_blocks=kv, arm=arm, method=LABELS[arm],
                    capacity_qps=group['capacity_qps'],
                    next_failed_qps=group['next_failed_qps'],
                    resolution_qps=group['resolution_qps'],
                    mean_ttft_ms=reference['summary']['mean_ttft_ms'],
                    archived_capacity_qps=group['archived_capacity_qps'],
                    windows=50, reference_qps=.5))
    write_csv(ANALYSIS / 'by_source.csv', by_source)
    paired = defaultdict(dict)
    for row in by_source:
        paired[row['kv_blocks'], row['arm']][row['source']] = row
    macro = []
    for kv in run.CAPACITIES:
        for arm in run.METHODS:
            pair = paired[kv, arm]
            assert set(pair) == set(SOURCES)
            macro.append(dict(
                kv_blocks=kv, arm=arm, method=LABELS[arm],
                mean_ttft_ms=sum(pair[s]['mean_ttft_ms'] for s in SOURCES)/2,
                mean_capacity_qps=sum(pair[s]['capacity_qps'] for s in SOURCES)/2,
                capacity_upper_qps=sum(pair[s]['next_failed_qps'] for s in SOURCES)/2,
                archived_capacity_qps=sum(pair[s]['archived_capacity_qps'] for s in SOURCES)/2,
                reference_qps=.5, windows_per_workload=50))
    write_csv(ANALYSIS / 'macro_averages.csv', macro)
    return by_source, macro


def comparisons(macro):
    result = []
    for kv in run.CAPACITIES:
        group = {p['arm']: p for p in macro if p['kv_blocks'] == kv}
        spare = group['arrival']
        dominators = [group[arm] for arm in run.METHODS[1:]
                      if dominates(group[arm], spare)]
        secure = [p for p in dominators
                  if p['mean_capacity_qps'] >= spare['capacity_upper_qps'] - 1e-9]
        best_baseline = max((group[arm] for arm in run.METHODS[1:]),
                            key=lambda p: p['mean_capacity_qps'])
        result.append(dict(
            kv_blocks=kv, spare_ttft_ms=spare['mean_ttft_ms'],
            spare_capacity_qps=spare['mean_capacity_qps'],
            spare_capacity_upper_qps=spare['capacity_upper_qps'],
            best_baseline_capacity_qps=best_baseline['mean_capacity_qps'],
            best_baseline_method=best_baseline['method'],
            dominated=bool(dominators),
            dominated_by='; '.join(p['method'] for p in dominators),
            dominated_beyond_qps_bracket='; '.join(p['method'] for p in secure),
            nondominated=not dominators))
    write_csv(ANALYSIS / 'same_kv_pareto.csv', result)
    return result


def legend():
    return [Line2D([], [], linestyle='None', marker=MARKERS[arm],
                   markersize=7 if arm == 'arrival' else 6,
                   markerfacecolor='white' if arm == 'arrival' else COLORS[arm],
                   markeredgecolor=COLORS[arm], markeredgewidth=1.5 if arm == 'arrival' else .7,
                   label='SPARE' if arm == 'arrival' else LABELS[arm])
            for arm in run.METHODS]


def draw_panel(ax, kv, macro, status):
    points = [p for p in macro if p['kv_blocks'] == kv]
    by_arm = {p['arm']: p for p in points}
    for arm in (*run.METHODS[1:], 'arrival'):
        p = by_arm[arm]
        ours = arm == 'arrival'
        ax.scatter(p['mean_ttft_ms'], p['mean_capacity_qps'],
                   marker=MARKERS[arm], s=65 if ours else 39,
                   facecolors='white' if ours else COLORS[arm],
                   edgecolors=COLORS[arm], linewidths=1.6 if ours else .8,
                   alpha=1 if ours else .85, zorder=5 if ours else 3)
    xs = [p['mean_ttft_ms'] for p in points]
    assert min(xs) > 0
    ax.set_xscale('log')
    ys = [p['mean_capacity_qps'] for p in points]
    ax.set_xlim(min(xs)*.92, max(xs)*1.1)
    padding = max(.12, (max(ys)-min(ys))*.12)
    ax.set_ylim(max(0, min(ys)-padding), max(ys)+padding)
    ax.set_title(f'KV {kv}  {"SPARE dominated" if status["dominated"] else "SPARE nondominated"}',
                 loc='left', color='#a72e30' if status['dominated'] else '#25383b',
                 fontsize=8, pad=6)
    ax.grid(axis='y', color='#dce2e5', linewidth=.45)
    ax.set_axisbelow(True)
    ticks = sorted({round(value / 10) * 10 for value in
                    (min(xs), math.sqrt(min(xs)*max(xs)), max(xs))})
    ax.set_xticks(ticks, labels=[str(value) for value in ticks])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def plot_grid(kvs, macro, statuses, stem, overview=False):
    rows, cols = (8, 4) if overview else (2, 4)
    width, height = (16, 21) if overview else (14.4, 7.2)
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(left=.07, right=.985, top=.975 if overview else .94,
                        bottom=.085 if overview else .20,
                        wspace=.26, hspace=.53 if overview else .36)
    for index, (ax, kv) in enumerate(zip(axes.flat, kvs)):
        draw_panel(ax, kv, macro, statuses[kv])
        if index // cols == rows-1:
            ax.set_xlabel('Mean TTFT (ms; 0.5 QPS)')
        if index % cols == 0:
            ax.set_ylabel('Capacity (QPS)')
    fig.legend(handles=legend(), loc='lower center', ncol=8,
               bbox_to_anchor=(.53, .01 if overview else .055), fontsize=7,
               columnspacing=1.0, handletextpad=.35)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=list(axes.flat),
                                       json_out=str(stem)+'.alignment.json', strict=True)
    fig.savefig(str(stem)+'.png', dpi=300)
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    plt.close(fig)


def report(by_source, macro, rows):
    dominated = [r for r in rows if r['dominated']]
    lines = [
        '# 32-KV, 50-window fine-QPS comparison', '',
        'CPU simulator only. Eight fixed implementations: SPARE and seven baselines.',
        'All methods use the same 0.5 QPS offered load for mean TTFT. Capacity is the',
        'largest passing offered QPS before the first failure, averaged equally over',
        'Code and Conversation. Each workload uses the same 50 test windows per method.',
        'Pass rule: all windows valid, mean P95 TTFT <= 12 s, mean P99 TBT <= 200 ms,',
        'mean growth in unfinished requests from 40 to 60 s <= max(2, offered QPS).',
        'Every per-workload passing/failing bracket is at most 0.02 QPS wide. The',
        'macro-average is a score across workloads, not a measured mixed-stream capacity.',
        'Dominance is computed only within the same KV, from point estimates.', '',
        f'SPARE dominated at {len(dominated)}/{len(rows)} KV levels: '
        + (', '.join(str(r['kv_blocks']) for r in dominated) or 'none') + '.', '',
        '|KV blocks|SPARE TTFT (ms)|SPARE capacity (QPS)|Best baseline capacity (QPS)|Dominated by|',
        '|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(f"|{row['kv_blocks']}|{row['spare_ttft_ms']:.2f}|"
                     f"{row['spare_capacity_qps']:.3f}|{row['best_baseline_capacity_qps']:.3f}|"
                     f"{row['dominated_by'] or '-'}|")
    lines += ['', 'The 0.02-QPS bracket is grid resolution, not statistical uncertainty.',
              'These 50-window test data were also used in the earlier coarse scan;',
              'the fine scan reuses its saved rate records. S3 and UniBoost are rule',
              'adaptations/proxies, not full original-system reproductions.']
    (ANALYSIS / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    summary = dict(kv_levels=len(run.CAPACITIES), methods=len(run.METHODS),
                   windows_per_workload=50, method_kv_points=len(macro),
                   source_rows=len(by_source), dominated_kv=[r['kv_blocks'] for r in dominated],
                   nondominated_kv=[r['kv_blocks'] for r in rows if not r['dominated']],
                   dominated_by={str(r['kv_blocks']): r['dominated_by'] for r in dominated},
                   all_brackets_le_0_02=all(r['resolution_qps'] <= .020001 for r in by_source))
    (ANALYSIS / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    by_source, macro = load()
    rows = comparisons(macro)
    statuses = {r['kv_blocks']: r for r in rows}
    plot_grid(run.CAPACITIES, macro, statuses, ANALYSIS / 'pareto_32kv_overview', overview=True)
    for index in range(4):
        kvs = run.CAPACITIES[index*8:(index+1)*8]
        plot_grid(kvs, macro, statuses, ANALYSIS / f'pareto_32kv_page{index+1}')
    print(report(by_source, macro, rows))


if __name__ == '__main__':
    main()
