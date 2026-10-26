"""Compare capacity and reference-load mean TTFT across KV sizes."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from research.www_capacity_pareto_20261015.run import OUT, read
from research.www_capacity_pareto_20261015.sensitivity import capacity, load_points
from research.www_pareto_papers_20261014.run import LABELS

FIGURES = OUT / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib_cache'))

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

SPARE = '#B83B3B'
FRONT = '#087E83'
OTHER = '#B5BDC2'
TEXT = '#49535A'
SCENES = (('code', 768), ('code', 4096),
          ('conversation', 768), ('conversation', 4096))
SETTINGS = {
    'original_5s': ('Original strict SLO', 5000.),
    'exploratory_12s': ('Exploratory SLO', 12000.),
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


def dominates(first, second):
    return (first['qps'] >= second['qps']
            and first['ttft_ms'] <= second['ttft_ms']
            and (first['qps'] > second['qps']
                 or first['ttft_ms'] < second['ttft_ms']))


def scene_rows(points, selection, sensitivity, source, blocks, setting, limit):
    if setting == 'original_5s':
        reference = next(row['ref_rate'] for row in selection['references']
                         if (row['source'], row['capacity'], row['slo'])
                         == (source, blocks, 'strict'))
        chosen_budget = next(row['budget'] for row in selection['configs']
                             if (row['source'], row['capacity'], row['slo'],
                                 row['arm'], row['weight'])
                             == (source, blocks, 'strict', 'arrival', .5))
    else:
        scene = next(row for row in sensitivity['scenes']
                     if (row['source'], row['blocks']) == (source, blocks))
        reference = scene['ref_rate']
        chosen_budget = scene['selected'][5]['budget']

    rows = []
    for (src, cap, arm, budget), point in points.items():
        if (src, cap) != (source, blocks):
            continue
        qps = (point['qps'] if setting == 'original_5s' else
               capacity(point, limit, 200., .05, 2.))
        rows.append(dict(setting=setting, source=source, kv_blocks=blocks,
                         arm=arm, budget=budget, reference_qps=reference,
                         qps=qps,
                         ttft_ms=point['by_rate'][str(reference)]['mean_ttft_ms'],
                         feasible=qps >= reference,
                         selected=arm == 'arrival' and budget == chosen_budget))
    assert len(rows) == 48 and sum(row['selected'] for row in rows) == 1
    return rows


def external_front(rows):
    external = [row for row in rows if row['arm'] != 'arrival' and row['feasible']]
    return [row for row in external
            if not any(dominates(other, row) for other in external if other is not row)]


def short_label(arm):
    return {
        'apt_serve_kv_rule': 'Apt-Serve KV',
        'past_future_multi': 'Past-Future multi',
        'past_future': 'Past-Future',
        'sola_calibrated': 'SOLA',
        's3_proxy': 'S3 proxy',
    }.get(arm, LABELS[arm])


def plot_scene(ax, rows, panel):
    visible = [row for row in rows if row['feasible']]
    chosen = next(row for row in rows if row['selected'])
    front = external_front(rows)
    front_ids = {id(row) for row in front}
    other = [row for row in visible if row['arm'] != 'arrival'
             and id(row) not in front_ids]
    spare_other = [row for row in visible if row['arm'] == 'arrival'
                   and not row['selected']]
    ax.scatter([row['ttft_ms'] for row in other], [row['qps'] for row in other],
               s=17, color=OTHER, alpha=.6, linewidths=0, zorder=2)
    ax.scatter([row['ttft_ms'] for row in spare_other],
               [row['qps'] for row in spare_other], s=30, facecolors='none',
               edgecolors=SPARE, linewidths=.9, alpha=.65, zorder=3)
    ax.scatter([row['ttft_ms'] for row in front], [row['qps'] for row in front],
               s=40, color=FRONT, marker='s', edgecolors='white',
               linewidths=.5, zorder=4)
    ax.scatter(chosen['ttft_ms'], chosen['qps'], s=72, color=SPARE,
               edgecolors='white', linewidths=.8, zorder=5)
    for row in front:
        offset_y = -19 if row['arm'] == 'past_future_multi' else -11
        ax.annotate(f'{short_label(row["arm"])} T{row["budget"]}',
                    (row['ttft_ms'], row['qps']), xytext=(6, offset_y),
                    textcoords='offset points', fontsize=5.8, color=FRONT)

    max_x = max(row['ttft_ms'] for row in visible)
    max_y = max(row['qps'] for row in visible)
    ax.set_xlim(0, max_x * 1.17)
    ax.set_ylim(0, max_y * 1.19)
    ax.grid(axis='y', color='#E6EAEC', linewidth=.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(f'Mean TTFT at {chosen["reference_qps"]:g} QPS (ms)')
    ax.set_ylabel('Serving capacity (QPS)')
    source = 'Code' if chosen['source'] == 'code' else 'Conversation'
    ax.set_title(f'{panel}   {source}  |  KV {chosen["kv_blocks"]}',
                 loc='left', fontweight='bold', pad=8)

    feasible_external = [row for row in visible if row['arm'] != 'arrival']
    if any(dominates(row, chosen) for row in feasible_external):
        status = 'SPARE dominated'
    elif all(dominates(chosen, row) or
             (chosen['qps'] == row['qps'] and chosen['ttft_ms'] == row['ttft_ms'])
             for row in feasible_external):
        status = 'SPARE dominates baselines'
    else:
        status = 'SPARE on Pareto front'
    ax.text(.98, .97, status, transform=ax.transAxes, ha='right', va='top',
            fontsize=6.5, color=SPARE if status != 'SPARE dominated' else TEXT)
    return dict(setting=chosen['setting'], source=chosen['source'],
                kv_blocks=chosen['kv_blocks'], reference_qps=chosen['reference_qps'],
                selected_budget=chosen['budget'], spare_qps=chosen['qps'],
                spare_ttft_ms=chosen['ttft_ms'], status=status,
                feasible_external=len(feasible_external),
                infeasible_candidates=len(rows)-len(visible),
                external_front='; '.join(
                    f'{short_label(row["arm"])} T{row["budget"]}: '
                    f'{row["qps"]:g} QPS, {row["ttft_ms"]:.0f} ms'
                    for row in front))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    points = load_points('test')
    selection = read(OUT / 'selection.json')
    sensitivity = next(row for row in read(OUT / 'slo_sensitivity.json')
                       if row['p95_ttft_ms'] == 12000 and row['p99_tbt_ms'] == 200)
    source_data = []
    summary = []
    for setting, (name, limit) in SETTINGS.items():
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))
        fig.subplots_adjust(left=.10, right=.97, bottom=.125, top=.82,
                            wspace=.31, hspace=.48)
        for ax, (source, blocks), panel in zip(axes.flat, SCENES, 'abcd'):
            rows = scene_rows(points, selection, sensitivity,
                              source, blocks, setting, limit)
            source_data.extend(rows)
            summary.append(plot_scene(ax, rows, panel))

        handles = [
            Line2D([], [], marker='o', linestyle='', markerfacecolor=SPARE,
                   markeredgecolor='white', markersize=7, label='SPARE selected'),
            Line2D([], [], marker='o', linestyle='', markerfacecolor='none',
                   markeredgecolor=SPARE, markersize=6, label='SPARE other budgets'),
            Line2D([], [], marker='s', linestyle='', markerfacecolor=FRONT,
                   markeredgecolor='white', markersize=6, label='External Pareto front'),
            Line2D([], [], marker='o', linestyle='', markerfacecolor=OTHER,
                   markeredgecolor=OTHER, markersize=5, label='Other external'),
        ]
        fig.legend(handles=handles, ncol=4, loc='upper center',
                   bbox_to_anchor=(.5, .905), columnspacing=1.1,
                   handletextpad=.4, fontsize=6.5)
        fig.suptitle(f'KV cache comparison | {name}: P95 TTFT <= {limit/1000:g} s',
                     x=.10, y=.975, ha='left', fontsize=10, fontweight='bold')
        fig.text(.10, .939, 'P99 TBT <= 200 ms; backlog limit = max(2, offered QPS).',
                 fontsize=6.5, color=TEXT)
        fig.text(.10, .035, 'CPU simulator, 50 test windows. Each panel uses one '
                 'validation-fixed reference load; 11 alpha values select one '
                 'SPARE budget per panel.', fontsize=6, color=TEXT)
        base = FIGURES / f'kv_capacity_pareto_{setting}'
        fig.canvas.draw()
        require_matplotlib_panel_alignment(
            fig, axes=list(axes.flat), json_out=str(base) + '.alignment.json',
            overlay_svg=str(base) + '.alignment.svg', strict=True)
        for suffix, kwargs in (('.png', {'dpi': 350}),
                               ('.pdf', {}), ('.svg', {}),
                               ('.tiff', {'dpi': 600})):
            fig.savefig(str(base) + suffix, **kwargs)
        plt.close(fig)
        print(base)

    write_csv(FIGURES / 'kv_capacity_pareto_source_data.csv', source_data)
    write_csv(FIGURES / 'kv_capacity_pareto_summary.csv', summary)
    for row in summary:
        print(f"{row['setting']} {row['source']} KV{row['kv_blocks']}: "
              f"{row['status']}; SPARE T{row['selected_budget']} "
              f"{row['spare_qps']:g} QPS / {row['spare_ttft_ms']:.0f} ms; "
              f"external front: {row['external_front']}")


if __name__ == '__main__':
    main()
