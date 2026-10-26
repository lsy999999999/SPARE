"""Figure B: paired KV capacity, latency and recovery-cost sensitivity."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAPACITY = ROOT / 'artifacts/www_kv_capacity_paper_20261015_r1'
OLD = ROOT / 'artifacts/www_kv_utilization_20261015_r1'
NEW = ROOT / 'artifacts/www_kv_degradation_20261016_r1'
OUT = NEW / 'analysis'
FIG = ROOT / 'writing/www/figures/newidea'
os.environ.setdefault('MPLCONFIGDIR', str(NEW / 'mpl_cache'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

CAPS = (768, 1024, 1536, 2048, 3072, 4096)
METHODS = (
    ('arrival', 'SPARE', '#195c9b', '-', 'o'),
    ('native', 'vLLM V1', '#494949', '--', 's'),
    ('sarathi', 'Sarathi rule', '#439bc2', '-.', '^'),
    ('sola_rule', 'SOLA rule', '#d25a35', ':', 'v'),
    ('uniboost_static', 'UniBoost rule', '#9562a0', '--', 'D'),
    ('past_future', 'Past-Future rule', '#238a83', '-.', 'P'),
    ('s3_proxy', 'S3 proxy', '#b64342', ':', 'X'),
    ('apt_serve_kv_rule', 'Apt-Serve rule', '#a67b1e', '--', '*'),
)
LABELS = {arm: label for arm, label, *_ in METHODS}


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load():
    OUT.mkdir(parents=True, exist_ok=True)
    capacity_design = json.loads((CAPACITY / 'design.json').read_text())
    old_design = json.loads((OLD / 'design.json').read_text())
    new_design = json.loads((NEW / 'design.json').read_text())
    assert capacity_design['default_budget'] == old_design['budget'] == new_design['budget'] == 1024
    assert capacity_design['chunk'] == old_design['chunk'] == new_design['chunk'] == 256
    assert capacity_design['concurrency'] == old_design['concurrency'] == new_design['concurrency'] == 32
    assert capacity_design['timing_sha256'] == old_design['timing_sha256'] == new_design['timing_sha256']
    assert set(CAPS) == set(capacity_design['capacities'])
    assert set(CAPS) == set(old_design['capacities']) | set(new_design['capacities'])
    assert set(arm for arm, *_ in METHODS) <= set(capacity_design['main_arms'])
    assert new_design['rate'] == old_design['rate'] == 4.
    assert new_design['horizon_ms'] == old_design['horizon_ms'] == 120000.
    old_samples = {s['name']: s for s in old_design['samples']}
    new_samples = {s['name']: s for s in new_design['samples']}
    paper_samples = {s['name']: s for s in capacity_design['samples']}
    assert old_samples == new_samples
    assert set(old_samples) <= set(paper_samples)
    for sample in old_samples.values():
        actual = hashlib.sha256((ROOT / sample['input']).read_bytes()).hexdigest()
        assert capacity_design['input_sha256'][sample['input']] == actual

    paper = read_csv(CAPACITY / 'analysis/by_source.csv')
    chosen = [r for r in paper if r['arm'] in LABELS]
    assert len(chosen) == 2 * len(CAPS) * len(METHODS)
    raw, aggregate = [], []
    request_hashes = defaultdict(set)
    for source in ('code', 'conversation'):
        samples = sorted((s for s in old_samples.values() if s['source'] == source),
                         key=lambda s: s['name'])
        assert len(samples) == 2
        for blocks in CAPS:
            root = OLD if blocks in old_design['capacities'] else NEW
            for arm, label, *_ in METHODS:
                point = next(r for r in chosen if (r['source'], int(r['kv_blocks']), r['arm']) ==
                             (source, blocks, arm))
                observed = []
                for sample in samples:
                    directory = root / 'runs' / f"{sample['name']}_c{blocks}_{arm}"
                    result = json.loads((directory / 'result.json').read_text())
                    request_hashes[sample['name']].add(result['request_sha256'])
                    with gzip.open(directory / 'iterations.jsonl.gz', 'rt') as f:
                        trace = [json.loads(line) for line in f]
                    assert len(trace) == result['rounds']
                    assert sum(r['preemptions'] for r in trace) == result['preemptions']
                    assert sum(r['recomputed_tokens'] for r in trace) == result['recomputed_tokens']
                    mid, end = result['pressure_windows']
                    assert mid['at_ms'] == 60000. and end['at_ms'] == 120000.
                    assert end['outstanding'] - mid['outstanding'] == (
                        end['arrivals'] - mid['arrivals'] - end['completed'] + mid['completed'])
                    arrivals = end['arrivals'] - mid['arrivals']
                    assert arrivals > 0
                    events = [r for r in trace if 60000. <= r['simulated_start_ms'] < 120000.]
                    row = dict(source=source, sample=sample['name'], kv_blocks=blocks,
                               arm=arm, method=label, offered_qps=4.,
                               arrivals_60_120=arrivals,
                               preemptions=sum(r['preemptions'] for r in events),
                               recomputed_tokens=sum(r['recomputed_tokens'] for r in events))
                    row['preemptions_per_100_arrivals'] = 100 * row['preemptions'] / arrivals
                    row['recomputed_tokens_per_arrival'] = row['recomputed_tokens'] / arrivals
                    raw.append(row)
                    observed.append(row)
                assert len(observed) == 2
                merged = dict(source=source, kv_blocks=blocks, arm=arm, method=label,
                              capacity_qps=float(point['capacity_qps']),
                              first_failed_qps=point['next_failed_qps'],
                              capacity_lower_bound=point['capacity_is_lower_bound'],
                              mean_ttft_ms_at_0_5_qps=float(point['mean_ttft_ms']),
                              capacity_windows=50, recovery_windows=2)
                for key in ('preemptions_per_100_arrivals', 'recomputed_tokens_per_arrival'):
                    values = [r[key] for r in observed]
                    merged[key] = float(np.mean(values))
                    merged[key + '_min'] = min(values)
                    merged[key + '_max'] = max(values)
                aggregate.append(merged)
    assert all(len(hashes) == 1 for hashes in request_hashes.values())
    assert len(raw) == 192 and len(aggregate) == 96
    write_csv(OUT / 'recovery_per_window.csv', raw)
    write_csv(OUT / 'figure_b_source_data.csv', aggregate)
    return aggregate


def plot(data):
    FIG.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'], 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'legend.frameon': False})
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    fig.subplots_adjust(left=.06, right=.985, bottom=.20, top=.92, wspace=.32, hspace=.48)
    panels = (
        ('capacity_qps', 'SLO serving capacity (QPS)'),
        ('mean_ttft_ms_at_0_5_qps', 'Mean TTFT at 0.5 QPS (ms)'),
        ('preemptions_per_100_arrivals', 'Preemptions / 100 arrivals'),
        ('recomputed_tokens_per_arrival', 'Recomputed tokens / arrival'),
    )
    for i, source in enumerate(('code', 'conversation')):
        for j, (field, ylabel) in enumerate(panels):
            ax = axes[i, j]
            for arm, label, color, linestyle, marker in METHODS:
                points = sorted((r for r in data if r['source'] == source and r['arm'] == arm),
                                key=lambda r: r['kv_blocks'])
                assert [r['kv_blocks'] for r in points] == list(CAPS)
                values = np.array([r[field] for r in points])
                opts = dict(label=label, color=color, linestyle=linestyle, marker=marker,
                            markersize=3.3, linewidth=1.8 if arm == 'arrival' else 1.,
                            alpha=1 if arm == 'arrival' else .84, zorder=5 if arm == 'arrival' else 2)
                if j >= 2:
                    lower = np.array([r[field + '_min'] for r in points])
                    upper = np.array([r[field + '_max'] for r in points])
                    ax.errorbar(range(len(CAPS)), values, yerr=[values-lower, upper-values],
                                elinewidth=.5, capsize=1.3, **opts)
                else:
                    ax.plot(range(len(CAPS)), values, **opts)
            ax.set_xticks(range(len(CAPS)), [str(v) for v in CAPS])
            ax.tick_params(axis='x', labelsize=6.7)
            ax.set_xlabel('KV capacity (blocks)')
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.set_title(source.capitalize())
            ax.grid(axis='y', color='#e1e4e6', linewidth=.5)
            ax.annotate(chr(97+i*4+j), xy=(0, 1), xycoords='axes fraction',
                        xytext=(-25, 12), textcoords='offset points',
                        fontweight='bold', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .03), ncol=4)
    fig.canvas.draw()
    stem = FIG / 'kv_capacity_degradation_b'
    require_matplotlib_panel_alignment(fig, json_out=str(stem)+'.alignment.json', strict=True)
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    fig.savefig(str(stem)+'.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    plot(load())
