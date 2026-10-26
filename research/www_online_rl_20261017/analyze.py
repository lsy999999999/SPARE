"""Audit and plot the paired online-RL versus SPARE capacity experiment."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_online_rl_mpl')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

from research.www_online_rl_20261017 import run
from research.www_kv_capacity_50wide_20261016.analyze import COLORS, LABELS, MARKERS
from research.www_kv_capacity_fine_20261016.same_qps_pareto import dominates

OUT = run.OUT
FIG = OUT / 'analysis'
PRIOR = run.prior.OUT / 'analysis' / 'macro_averages.csv'
RL_COLOR = '#236b90'

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
    'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
})


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load():
    status = run.base.read(OUT / 'evaluation_status.json')
    assert status['complete'] and status['groups'] == 2 * len(run.EVAL_KV)
    old = list(csv.DictReader(PRIOR.open()))
    baseline = {}
    for row in old:
        kv = int(row['kv_blocks'])
        if kv in run.EVAL_KV:
            baseline[kv, row['arm']] = dict(kv_blocks=kv, arm=row['arm'],
                method=row['method'], mean_ttft_ms=float(row['mean_ttft_ms']),
                mean_capacity_qps=float(row['mean_capacity_qps']))
    by_source, macro = [], []
    for kv in run.EVAL_KV:
        source_rows = []
        for source in ('code', 'conversation'):
            group = run.base.read(OUT / 'groups' / f'{source}_c{kv}_online_rl.json')
            assert group['resolution_qps'] <= .020001
            path = OUT / 'rates' / f'{source}_c{kv}_online_rl_r0.5.json'
            record = run.base.read(path)
            assert len(record['rows']) == 50
            assert record['summary'] == run.prior.old.summary(record['rows'], .5)
            capacity_rate = group['capacity_qps']
            capacity_record = run.base.read(OUT / 'rates' /
                f'{source}_c{kv}_online_rl_r{capacity_rate:g}.json')
            assert capacity_record['summary']['passed']
            reference_actions = [sum(r['rl_actions']) for r in record['rows']]
            capacity_actions = [sum(r['rl_actions']) for r in capacity_record['rows']]
            row = dict(source=source, kv_blocks=kv, arm='online_rl',
                       capacity_qps=group['capacity_qps'],
                       next_failed_qps=group['next_failed_qps'],
                       mean_ttft_ms=record['summary']['mean_ttft_ms'],
                       mean_schedule_ms=sum(r['schedule_mean_ms'] for r in record['rows']) / 50,
                       mean_wall_s=sum(r['wall_s'] for r in record['rows']) / 50,
                       reference_spare_action_fraction=sum(r['rl_actions'][0] for r in record['rows']) /
                           sum(reference_actions),
                       capacity_spare_action_fraction=sum(r['rl_actions'][0] for r in capacity_record['rows']) /
                           sum(capacity_actions))
            by_source.append(row)
            source_rows.append(row)
        macro.append(dict(kv_blocks=kv, arm='online_rl', method='Online RL',
                          mean_ttft_ms=sum(r['mean_ttft_ms'] for r in source_rows)/2,
                          mean_capacity_qps=sum(r['capacity_qps'] for r in source_rows)/2,
                          capacity_upper_qps=sum(r['next_failed_qps'] for r in source_rows)/2,
                          mean_schedule_ms=sum(r['mean_schedule_ms'] for r in source_rows)/2,
                          reference_spare_action_fraction=sum(
                              r['reference_spare_action_fraction'] for r in source_rows)/2,
                          capacity_spare_action_fraction=sum(
                              r['capacity_spare_action_fraction'] for r in source_rows)/2))
    write_csv(FIG / 'rl_by_source.csv', by_source)
    write_csv(FIG / 'rl_macro_averages.csv', macro)
    return baseline, macro


def compare(baseline, macro):
    rows = []
    for rl in macro:
        kv = rl['kv_blocks']
        spare = baseline[kv, 'arrival']
        all_points = [baseline[kv, arm] for arm in run.prior.METHODS]
        best = max(all_points[1:], key=lambda p: p['mean_capacity_qps'])
        rows.append(dict(kv_blocks=kv,
                         rl_ttft_ms=rl['mean_ttft_ms'], spare_ttft_ms=spare['mean_ttft_ms'],
                         rl_capacity_qps=rl['mean_capacity_qps'],
                         rl_capacity_upper_qps=rl['capacity_upper_qps'],
                         spare_capacity_qps=spare['mean_capacity_qps'],
                         best_prior_capacity_qps=best['mean_capacity_qps'],
                         best_prior_method=best['method'],
                         rl_dominates_spare=dominates(rl, spare),
                         spare_dominates_rl=dominates(spare, rl),
                         rl_nondominated=not any(dominates(p, rl) for p in all_points)))
    write_csv(FIG / 'same_kv_comparison.csv', rows)
    return rows


def legend():
    return [Line2D([], [], linestyle='None', marker='D', markersize=6,
                   markerfacecolor=RL_COLOR, markeredgecolor=RL_COLOR,
                   label='Online RL'),
            Line2D([], [], linestyle='None', marker='o', markersize=7,
                   markerfacecolor='white', markeredgecolor=COLORS['arrival'],
                   markeredgewidth=1.5, label='SPARE'),
            Line2D([], [], linestyle='None', marker='s', markersize=5,
                   markerfacecolor='#9aabb0', markeredgecolor='#9aabb0',
                   label='Other rules')]


def plot_page(kvs, baseline, macro_by_kv, page, *, main=False):
    rows = 2 if main or len(kvs) == 8 else 1
    cols = 2 if main else (4 if rows == 2 else len(kvs))
    size = (7.2, 5.5) if main else ((14.4, 7.2) if rows == 2 else (14.4, 3.8))
    fig, axes = plt.subplots(rows, cols, figsize=size,
                             squeeze=False)
    fig.subplots_adjust(left=.10 if main else .07, right=.98 if main else .985,
                        top=.94 if rows == 2 else .89,
                        bottom=.20 if rows == 2 else .29, wspace=.28, hspace=.36)
    for index, ax in enumerate(axes.flat):
        if index >= len(kvs):
            ax.set_visible(False)
            continue
        kv = kvs[index]
        points = [baseline[kv, arm] for arm in run.prior.METHODS]
        rl = macro_by_kv[kv]
        for point in points[1:]:
            ax.scatter(point['mean_ttft_ms'], point['mean_capacity_qps'],
                       marker='s', s=28, color='#9aabb0', alpha=.8, zorder=2)
        spare = points[0]
        ax.scatter(spare['mean_ttft_ms'], spare['mean_capacity_qps'],
                   marker='o', s=65, facecolors='white',
                   edgecolors=COLORS['arrival'], linewidths=1.5, zorder=5)
        ax.scatter(rl['mean_ttft_ms'], rl['mean_capacity_qps'],
                   marker='D', s=47, color=RL_COLOR, zorder=6)
        xs = [p['mean_ttft_ms'] for p in points] + [rl['mean_ttft_ms']]
        ys = [p['mean_capacity_qps'] for p in points] + [rl['mean_capacity_qps']]
        assert min(xs) > 0
        ax.set_xscale('log')
        ax.set_xlim(min(xs)*.92, max(xs)*1.1)
        pad = max(.12, (max(ys)-min(ys))*.12)
        ax.set_ylim(max(0, min(ys)-pad), max(ys)+pad)
        ax.set_title(f'KV {kv}', loc='left')
        ax.grid(axis='y', color='#dce2e5', linewidth=.45)
        ax.set_axisbelow(True)
        ticks = sorted({round(value / 10) * 10 for value in
                        (min(xs), math.sqrt(min(xs)*max(xs)), max(xs))})
        ax.set_xticks(ticks, labels=[str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        if index >= cols * (rows-1):
            ax.set_xlabel('Mean TTFT (ms; 0.5 QPS)')
        if index % cols == 0:
            ax.set_ylabel('Capacity (QPS)')
    fig.legend(handles=legend(), loc='lower center', ncol=3,
               bbox_to_anchor=(.53, .06 if rows == 2 else .07), fontsize=7)
    fig.canvas.draw()
    stem = FIG / ('online_rl_main' if main else f'online_rl_same_kv_page{page}')
    require_matplotlib_panel_alignment(fig, axes=[ax for ax in axes.flat if ax.get_visible()],
                                       json_out=str(stem)+'.alignment.json', strict=True)
    fig.savefig(str(stem)+'.png', dpi=300)
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    plt.close(fig)


def report(rows):
    win = [r for r in rows if r['rl_dominates_spare']]
    lose = [r for r in rows if r['spare_dominates_rl']]
    non = [r for r in rows if r['rl_nondominated']]
    checkpoint = run.base.read(OUT / 'checkpoint.json')
    validation = run.base.read(OUT / 'cost_validation.json')
    convergence = run.base.read(OUT / 'convergence_status.json')
    cross_host = run.base.read(OUT / 'cross_host_audit.json')
    macro = list(csv.DictReader((FIG / 'rl_macro_averages.csv').open()))
    reference_fallback = sum(float(r['reference_spare_action_fraction']) for r in macro)/len(macro)
    capacity_fallback = sum(float(r['capacity_spare_action_fraction']) for r in macro)/len(macro)
    lines = ['# Online RL versus SPARE', '',
             'Same V1 CPU simulator, fitted clock, 50 paired test windows per workload.',
             'The controller is tabular Q-learning over online state and selects a',
             'saving threshold for the same four guarded SPARE candidate plans.',
             'It is a learned threshold controller, not a full-action PPO scheduler.',
             'Train: 16 development windows per workload, 4 KV levels, 4 QPS rates;',
             'remaining validation windows and all test windows were excluded from training.',
             'Test: KV >= 1024, mean TTFT at 0.5 QPS; capacity uses the same',
             'P95 TTFT <= 12 s, P99 TBT <= 200 ms, backlog-drift <= max(2,QPS)',
             'and <=0.02-QPS capacity bracket as the existing figure.', '',
             f'Training: {checkpoint["training_episodes"]} episodes, '
             f'{checkpoint["updates"]} TD updates, {checkpoint["wall_s"]:.1f} s wall, '
             f'{checkpoint["cpu_s"]:.1f} s CPU; GPU training: no.',
             f'Training failures: {len(checkpoint["failures"])}.',
             f'SPARE fallback action fraction: {reference_fallback:.3f} at reference QPS, '
             f'{capacity_fallback:.3f} at each RL capacity boundary.',
             'Held-out validation schedule cost, mean over eight windows per point:',
             *[f'KV {v["kv_blocks"]}, {v["rate"]:g} QPS: SPARE '
               f'{v["mean_spare_schedule_ms"]:.3f} ms/round, RL '
               f'{v["mean_rl_schedule_ms"]:.3f} ms/round.'
               for v in validation['summary']],
             f'RL dominates SPARE: {len(win)}/{len(rows)} KV; '
             f'SPARE dominates RL: {len(lose)}/{len(rows)} KV; '
             f'RL nondominated among old methods: {len(non)}/{len(rows)} KV.', '',
             f'Cross-host audit: {len(cross_host["checked"])}/17 overlapping KV groups, '
             f'{len(cross_host["differences"])} outcome differences.',
             f'Learning curves: {len(convergence["seeds"])} seeds, '
             'non-monotonic held-out scores through 512 episodes; no convergence claim.',
             'All three final policies are slower than SPARE at KV 4096 and 4 QPS.',
             'See [convergence report](CONVERGENCE_REPORT.md) and '
             '[learning curve](online_rl_convergence.pdf).', '',
             'Figures: [main four-KV comparison](online_rl_main.pdf); '
             '[all KV page 1](online_rl_same_kv_page1.pdf), '
             '[page 2](online_rl_same_kv_page2.pdf), '
             '[page 3](online_rl_same_kv_page3.pdf), '
             '[page 4](online_rl_same_kv_page4.pdf).',
             'Source data: [per-workload RL values](rl_by_source.csv), '
             '[macro averages](rl_macro_averages.csv), '
             '[same-KV comparison](same_kv_comparison.csv).', '',
             '|KV|RL TTFT ms|SPARE TTFT ms|RL capacity|SPARE capacity|Result|',
             '|---:|---:|---:|---:|---:|---|']
    for r in rows:
        status = ('RL dominates' if r['rl_dominates_spare'] else
                  'SPARE dominates' if r['spare_dominates_rl'] else 'tradeoff/tie')
        lines.append(f'|{r["kv_blocks"]}|{r["rl_ttft_ms"]:.2f}|{r["spare_ttft_ms"]:.2f}|'
                     f'{r["rl_capacity_qps"]:.3f}|{r["spare_capacity_qps"]:.3f}|{status}|')
    lines += ['', 'Capacity is the equal mean of separate Code and Conversation',
              'capacities, not a measured mixed-stream capacity. Results are point',
              'estimates from a CPU simulator; training and actor CPU time are',
              'reported separately from simulated request latency.']
    (FIG / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:21]))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    baseline, macro = load()
    rows = compare(baseline, macro)
    rl = {r['kv_blocks']: r for r in macro}
    plot_page((1024, 2048, 4096, 8192), baseline, rl, 0, main=True)
    for index in range(0, len(run.EVAL_KV), 8):
        plot_page(run.EVAL_KV[index:index+8], baseline, rl, index//8+1)
    report(rows)


if __name__ == '__main__':
    main()
