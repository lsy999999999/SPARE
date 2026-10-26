"""Exploratory TTFT-capacity scatter from the completed full-request KV scan."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/www_kv_extreme_stress_20261016_r1/analysis'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts/www_kv_extreme_stress_20261016_r1/mpl_cache'))

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

from research.www_kv_extreme_stress_20261016.analyze import STYLE
from research.www_kv_extreme_stress_20261016.run import CAPACITIES, METHODS

SOURCE = OUT / 'full_by_source.csv'
STEM = OUT / 'full_macro_ttft_capacity'
RED = '#bd3638'
COLORS = {
    'native': '#5c656b', 'sarathi': '#829da8', 'sola_rule': '#b17a48',
    'uniboost_static': '#897698', 'past_future': '#298a86',
    's3_proxy': '#bc8567', 'apt_serve_kv_rule': '#9c9d5b',
}
MARKERS = dict(zip((a for a in METHODS if a != 'arrival'), ('s', '^', 'v', 'D', 'P', 'X', '*')))
LABELS = {arm: label for arm, label, *_ in STYLE}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def load_means() -> list[dict]:
    with SOURCE.open() as stream:
        raw = list(csv.DictReader(stream))
    assert len(raw) == 2 * len(CAPACITIES['full']) * len(METHODS)
    pairs = defaultdict(dict)
    for row in raw:
        key = (int(row['kv_blocks']), row['arm'])
        assert row['source'] not in pairs[key]
        assert int(row['measured_windows']) == 10
        pairs[key][row['source']] = row

    means = []
    for kv in CAPACITIES['full']:
        for arm in METHODS:
            pair = pairs[kv, arm]
            assert set(pair) == {'code', 'conversation'}
            rows = [pair[source] for source in ('code', 'conversation')]
            means.append(dict(
                kv_blocks=kv, arm=arm, method=LABELS[arm],
                mean_ttft_ms=sum(float(r['mean_ttft_ms_at_0_5_qps']) for r in rows) / 2,
                mean_capacity_qps=sum(float(r['capacity_qps']) for r in rows) / 2,
                reference_feasible=all(r['reference_feasible'] == 'True' for r in rows),
                windows_per_workload=10,
            ))
    return means


def dominates(first: dict, second: dict) -> bool:
    return (first['reference_feasible'] and second['reference_feasible']
            and first['mean_ttft_ms'] <= second['mean_ttft_ms'] + 1e-8
            and first['mean_capacity_qps'] >= second['mean_capacity_qps'] - 1e-8
            and (first['mean_ttft_ms'] < second['mean_ttft_ms'] - 1e-8
                 or first['mean_capacity_qps'] > second['mean_capacity_qps'] + 1e-8))


def comparisons(means: list[dict]) -> list[dict]:
    result = []
    for kv in CAPACITIES['full']:
        ours = next(p for p in means if p['kv_blocks'] == kv and p['arm'] == 'arrival')
        rivals = [p for p in means if p['kv_blocks'] == kv and p['arm'] != 'arrival']
        result.append(dict(
            kv_blocks=kv, spare_ttft_ms=ours['mean_ttft_ms'],
            spare_capacity_qps=ours['mean_capacity_qps'],
            dominating_methods='; '.join(p['method'] for p in rivals if dominates(p, ours)),
            best_baseline_capacity_qps=max(p['mean_capacity_qps'] for p in rivals),
        ))
    return result


def plot(means: list[dict]) -> None:
    points = [p for p in means if p['reference_feasible']]
    assert all(p['mean_ttft_ms'] > 0 for p in points)
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7,
        'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    fig.subplots_adjust(left=.105, right=.97, top=.92, bottom=.27)

    for arm in METHODS:
        series = [p for p in points if p['arm'] == arm]
        if arm == 'arrival':
            continue
        ax.scatter([p['mean_ttft_ms'] for p in series],
                   [p['mean_capacity_qps'] for p in series],
                   marker=MARKERS[arm], s=26, color=COLORS[arm],
                   linewidth=.3, edgecolor='white', alpha=.57, zorder=2)

    ours = [p for p in points if p['arm'] == 'arrival']
    ax.scatter([p['mean_ttft_ms'] for p in ours],
               [p['mean_capacity_qps'] for p in ours],
               marker='o', s=80, facecolor='none', edgecolor=RED,
               linewidth=1.5, zorder=5)
    labels = {
        2048: ('2048', (11, 11)),
        8192: ('3072-8192', (11, 11)),
    }
    for kv, (label, offset) in labels.items():
        point = next(p for p in ours if p['kv_blocks'] == kv)
        ax.annotate(label, (point['mean_ttft_ms'], point['mean_capacity_qps']),
                    xytext=offset, textcoords='offset points', color=RED,
                    fontsize=7, va='center', zorder=6)

    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator((300, 500, 1000, 2000, 5000)))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(260, 7400)
    ax.set_ylim(0, 4.45)
    ax.set_xlabel('Mean TTFT at 0.5 QPS (ms; log scale)')
    ax.set_ylabel('Mean serving capacity (QPS)')
    ax.set_title('Code + Conversation, equal-weight average', loc='left', fontsize=9)
    ax.grid(axis='y', color='#e5e8e9', linewidth=.55)
    ax.set_axisbelow(True)

    handles = [Line2D([], [], linestyle='None', marker='o', markersize=7,
                      markerfacecolor='none', markeredgecolor=RED,
                      markeredgewidth=1.5, label='SPARE')]
    handles += [Line2D([], [], linestyle='None', marker=MARKERS[a], markersize=5,
                       markerfacecolor=COLORS[a], markeredgecolor='white',
                       label=LABELS[a]) for a in METHODS if a != 'arrival']
    fig.legend(handles=handles, ncol=4, loc='lower center',
               bbox_to_anchor=(.53, .055), fontsize=7, columnspacing=1.6)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=[ax],
                                       json_out=str(STEM) + '.alignment.json', strict=True)
    fig.savefig(str(STEM) + '.pdf')
    fig.savefig(str(STEM) + '.svg')
    fig.savefig(str(STEM) + '.png', dpi=350)
    plt.close(fig)


def main() -> None:
    means = load_means()
    result = comparisons(means)
    write_csv(OUT / 'full_macro_ttft_capacity.csv', means)
    write_csv(OUT / 'full_macro_dominance.csv', result)
    plot(means)
    print(f'{len(means)} source points, {sum(p["reference_feasible"] for p in means)} plotted; '
          f'{sum(not r["dominating_methods"] for r in result)}/{len(result)} SPARE points non-dominated')


if __name__ == '__main__':
    main()
