"""Paired first-token latency decomposition from complete scheduling traces."""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import run, trace

SOURCE = 'conversation'
KV = 2048
RATE = 3.
WARM_MS = 20000.
HORIZON_MS = 60000.
OUT = run.OUT / 'analysis' / 'ttft_breakdown'
PHASES = ('before_first_schedule_ms', 'between_prefill_rounds_ms',
          'scheduled_rounds_ms')


def archived_rows(arm):
    path = run.prior.OUT / 'rates' / f'{SOURCE}_c{KV}_{arm}_r{RATE:g}.json'
    record = run.prior.base.read(path)
    return {r['name']: r for r in record['rows']}


def decompose(result, log_path):
    records = {r['request_id']: r for r in result['probe_records']
               if WARM_MS <= r['arrival_ms'] < HORIZON_MS}
    assert records
    active = dict.fromkeys(records, 0.)
    with gzip.open(log_path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            duration = row['simulated_step_ms']
            for rid in row['scheduled']:
                if rid in records and row['simulated_start_ms'] < records[rid]['first_token_ms']:
                    active[rid] += duration
    pieces = []
    for rid, record in records.items():
        first_wait = record['scheduling_delay_ms']
        scheduled = active[rid]
        between = record['ttft_ms'] - first_wait - scheduled
        assert scheduled > 0 and between >= -1e-6, (rid, between)
        pieces.append(dict(request_id=rid, ttft_ms=record['ttft_ms'],
                           before_first_schedule_ms=first_wait,
                           between_prefill_rounds_ms=max(0., between),
                           scheduled_rounds_ms=scheduled))
    return pieces


def process_pair(sample):
    expected = {arm: archived_rows(arm)[sample['name']]
                for arm in ('spare', 'srp')}
    path = OUT / 'windows' / f'{sample["name"]}.json'
    if path.exists():
        saved = run.prior.base.read(path)
        assert saved['source'] == SOURCE and saved['kv_blocks'] == KV
        return saved
    rows = []
    summaries = {}
    for arm in ('spare', 'srp'):
        folder = OUT / 'runs' / sample['name'] / arm
        result = trace.replay(sample, arm, folder)
        assert result['trajectory_sha256'] == expected[arm]['trajectory_sha256']
        measured = run.prior.base.probe(result)
        for metric in ('mean_ttft_ms', 'p95_ttft_ms', 'p99_tbt_ms',
                       'backlog_drift', 'preemptions', 'recomputed_tokens'):
            assert measured[metric] == expected[arm][metric], (sample['name'], arm, metric)
        pieces = decompose(result, folder / 'iterations.jsonl.gz')
        assert math.isclose(np.mean([r['ttft_ms'] for r in pieces]),
                            expected[arm]['mean_ttft_ms'], abs_tol=1e-6)
        for piece in pieces:
            rows.append(dict(sample=sample['name'], arm=arm, **piece))
        summaries[arm] = dict(n_requests=len(pieces),
                              **{key: float(np.mean([r[key] for r in pieces]))
                                 for key in (*PHASES, 'ttft_ms')})
        assert math.isclose(sum(summaries[arm][key] for key in PHASES),
                            summaries[arm]['ttft_ms'], abs_tol=1e-6)
    assert summaries['spare']['n_requests'] == summaries['srp']['n_requests']
    path.parent.mkdir(parents=True, exist_ok=True)
    run.prior.base.write(path, dict(source=SOURCE, kv_blocks=KV, rate_qps=RATE,
                                   sample=sample['name'], phases=summaries,
                                   requests=rows))
    return run.prior.base.read(path)


def collect(workers):
    frozen = run.prior.prepare()
    samples = [s for s in frozen['samples'] if s['source'] == SOURCE]
    assert len(samples) == 50
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=run.prior.initialize) as pool:
        futures = {pool.submit(process_pair, sample): sample['name'] for sample in samples}
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 10 == 0:
                print(f'paired windows: {len(results)}/50', flush=True)
    return sorted(results, key=lambda row: row['sample'])


def summarize(windows):
    assert len(windows) == 50 and len({r['sample'] for r in windows}) == 50
    record = {}
    for arm in ('spare', 'srp'):
        record[arm] = {key: float(np.mean([w['phases'][arm][key] for w in windows]))
                       for key in (*PHASES, 'ttft_ms')}
        archived = run.prior.base.read(
            run.prior.OUT / 'rates' / f'{SOURCE}_c{KV}_{arm}_r{RATE:g}.json')
        assert math.isclose(record[arm]['ttft_ms'],
                            archived['summary']['mean_ttft_ms'], abs_tol=1e-6)
    delta = np.array([[w['phases']['srp'][key] - w['phases']['spare'][key]
                       for key in (*PHASES, 'ttft_ms')] for w in windows])
    rng = np.random.default_rng(20261017)
    draws = rng.integers(0, len(windows), size=(10000, len(windows)))
    bootstrap = delta[draws].mean(axis=1)
    paired = {key: dict(mean_ms=float(delta[:, i].mean()),
                        ci95_ms=[float(x) for x in np.percentile(bootstrap[:, i], [2.5, 97.5])])
              for i, key in enumerate((*PHASES, 'ttft_ms'))}
    request_rows = [row for window in windows for row in window['requests']]
    with (OUT / 'per_request.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(request_rows[0]))
        writer.writeheader()
        writer.writerows(request_rows)
    with (OUT / 'per_window.csv').open('w', newline='') as stream:
        fields = ['sample', 'arm', 'n_requests', *PHASES, 'ttft_ms']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for window in windows:
            for arm in ('spare', 'srp'):
                writer.writerow(dict(sample=window['sample'], arm=arm,
                                     **window['phases'][arm]))
    summary = dict(source=SOURCE, kv_blocks=KV, offered_qps=RATE,
                   windows=50, measured_arrivals_ms=[WARM_MS, HORIZON_MS],
                   phase_definition={
                       'before_first_schedule_ms': 'arrival to first scheduled round start',
                       'between_prefill_rounds_ms': 'time outside scheduled rounds after first service and before first token',
                       'scheduled_rounds_ms': 'whole duration of rounds containing this request before first token; includes concurrent batch work'},
                   per_window_equal_weight_means=record,
                   paired_srp_minus_spare=paired,
                   uncertainty='10,000 paired-window bootstrap resamples; exploratory windows')
    run.prior.base.write(OUT / 'summary.json', summary)
    return summary


def plot(summary):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_ttft_breakdown_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False})
    labels = {'before_first_schedule_ms': 'Before first scheduling',
              'between_prefill_rounds_ms': 'Between prefill rounds',
              'scheduled_rounds_ms': 'Rounds containing prefill'}
    colors = {'before_first_schedule_ms': '#777F89',
              'between_prefill_rounds_ms': '#C56A48',
              'scheduled_rounds_ms': '#397E90'}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=.11, right=.985, top=.86, bottom=.36, wspace=.47)
    ax = axes[0]
    for y, arm in enumerate(('spare', 'srp')):
        left = 0.
        for key in PHASES:
            value = summary['per_window_equal_weight_means'][arm][key] / 1000
            ax.barh(y, value, left=left, height=.48, color=colors[key],
                    label=labels[key] if y == 0 else None)
            left += value
        ax.text(left+.07, y, f'{left:.2f} s', va='center', fontsize=8)
    ax.set_yticks([0, 1], ['SPARE', 'SRP'])
    ax.invert_yaxis()
    ax.set_xlim(0, max(summary['per_window_equal_weight_means'][a]['ttft_ms']
                       for a in ('spare', 'srp'))/1000*1.27)
    ax.set_xlabel('Mean TTFT (s)')
    ax.set_title('(a) Mean TTFT decomposition', loc='left')
    ax.grid(axis='x', color='#E1E4E6', lw=.5)
    ax.set_axisbelow(True)
    handles, legend_labels = ax.get_legend_handles_labels()

    ax = axes[1]
    keys = (*PHASES, 'ttft_ms')
    names = ['First wait', 'Inter-chunk wait', 'Scheduled rounds', 'Total TTFT']
    y = np.arange(len(keys))
    means = np.array([summary['paired_srp_minus_spare'][k]['mean_ms']/1000 for k in keys])
    ci = np.array([summary['paired_srp_minus_spare'][k]['ci95_ms'] for k in keys])/1000
    ax.axvline(0, color='#555555', lw=.8)
    ax.errorbar(means, y, xerr=np.stack([means-ci[:, 0], ci[:, 1]-means]),
                fmt='o', color='#0F4D92', ecolor='#0F4D92', ms=4, capsize=2,
                lw=1.1)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel('SRP minus SPARE (s)')
    ax.set_title('(b) Paired SRP - SPARE', loc='left')
    ax.grid(axis='x', color='#E1E4E6', lw=.5)
    ax.set_axisbelow(True)
    fig.legend(handles, legend_labels, loc='lower center', ncol=3,
               bbox_to_anchor=(.5, .10), fontsize=7, columnspacing=1.2)
    fig.text(.5, .035, 'Conversation | 2048 KV blocks | 3 offered QPS | 50 paired windows; 95% bootstrap intervals',
             ha='center', fontsize=7)
    audit_script = Path.home() / '.codex/skills/nature-figure/scripts/audit_panel_alignment.py'
    if audit_script.is_file():
        spec = importlib.util.spec_from_file_location('audit_panel_alignment', audit_script)
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        fig.canvas.draw()
        audit.require_matplotlib_panel_alignment(
            fig, json_out=OUT / 'ttft_breakdown.alignment-audit.json')
    fig.savefig(OUT / 'ttft_breakdown.pdf', facecolor='white')
    fig.savefig(OUT / 'ttft_breakdown.svg', facecolor='white')
    fig.savefig(OUT / 'ttft_breakdown.png', dpi=600, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'plot'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.action == 'run':
        windows = collect(args.workers)
        summary = summarize(windows)
    else:
        summary = run.prior.base.read(OUT / 'summary.json')
        plot(summary)
    print(json.dumps(summary['per_window_equal_weight_means'], indent=2))


if __name__ == '__main__':
    main()
