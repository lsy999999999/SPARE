"""A real same-input scheduling timeline at a serving-capacity boundary."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_biobjective_trace_mpl')
import numpy as np

from . import run

SOURCE = 'conversation'
KV = 2048
RATE = 3.


def select_sample():
    def rows(arm):
        path = run.prior.OUT / 'rates' / f'{SOURCE}_c{KV}_{arm}_r{RATE:g}.json'
        return {r['name']: r for r in run.prior.base.read(path)['rows']}
    spare, srp = rows('spare'), rows('srp')
    assert spare.keys() == srp.keys() and len(spare) == 50
    ranked = sorted((srp[name]['mean_ttft_ms']-spare[name]['mean_ttft_ms'], name)
                    for name in spare)
    name = ranked[len(ranked)//2][1]
    frozen = run.prior.prepare()
    return next(s for s in frozen['samples'] if s['name'] == name), spare[name], srp[name]


def replay(sample, arm, folder):
    base = run.prior.base
    seed = int.from_bytes(__import__('hashlib').sha256(sample['name'].encode()).digest()[:4], 'big')
    requests = base.generate(sample, RATE)
    capture = {}

    def make(*args, **kwargs):
        scheduler, clock = base.ORIGINAL_MAKE(*args, **kwargs)
        capture['scheduler'] = scheduler
        return scheduler, clock

    class Clock:
        def estimate(self, batch):
            duration, details = base.TIMING.estimate(batch)
            capture['scheduler'].round_rows[-1]['step_ms'] = duration
            return duration, details

    with patch.object(base.SIM, 'make_scheduler', make):
        return base.SIM.simulate(
            requests, arm='arrival' if arm == 'spare' else 'srp',
            capacity=KV, budget=1024, chunk=256,
            history=base.HISTORY[SOURCE], seed=seed,
            fitted_timing=Clock(), capture_probe=True,
            output=folder, full_trace=True, max_wall_s=180)


def iterations(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def first_divergence(spare, srp):
    for left, right in zip(spare, srp):
        if left['scheduled'] != right['scheduled']:
            assert left['simulated_start_ms'] == right['simulated_start_ms']
            assert left['free_before'] == right['free_before']
            assert left['baseline'] == right['scheduled']
            return left, right
    raise AssertionError('No scheduling divergence')


def waiting_series(records, grid):
    arrivals = np.array([r['arrival_ms'] for r in records])
    first = np.array([r['first_token_ms'] for r in records])
    return np.array([np.count_nonzero((arrivals <= t) & (first > t)) for t in grid])


def plot(spare_run, srp_run, spare_rows, srp_rows, choice, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    left, right = choice
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 8, 'axes.titlesize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .75, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    fig.subplots_adjust(left=.1, right=.97, top=.92, bottom=.13,
                        hspace=.42, wspace=.27)
    colors = {'spare': '#0F4D92', 'srp': '#B64342'}
    plan = left['baseline']
    action = left['scheduled']
    state = {r['rid']: r for r in left['state']}
    chosen = [rid for rid in plan if plan[rid] > 0 and
              plan[rid] != action.get(rid, 0)]
    preserved_decode = sum(state[rid]['outputs'] > 0 and
                           state[rid]['tokens']-state[rid]['computed'] == 1 and
                           action.get(rid, 0) == plan[rid]
                           for rid in plan if plan[rid] > 0)
    if not chosen:
        chosen = list(plan)[:9]
    y = np.arange(len(chosen))
    axes[0, 0].barh(y-.18, [plan[rid] for rid in chosen], height=.34,
                    color=colors['srp'], label='Base plan / SRP')
    axes[0, 0].barh(y+.18, [action.get(rid, 0) for rid in chosen], height=.34,
                    color=colors['spare'], label='SPARE')
    axes[0, 0].set_yticks(y, [rid.rsplit(':', 1)[-1] for rid in chosen])
    axes[0, 0].set_ylim(len(chosen)+1.6, -.6)
    axes[0, 0].set_xlabel('Tokens scheduled in the same iteration')
    axes[0, 0].set_title('(a) First decision that differs', loc='left')
    axes[0, 0].text(.98, .05,
                    f'{preserved_decode} decode quotas kept\n'
                    f'New KV: {left["baseline_growth_blocks"]} to '
                    f'{left["actual_growth_blocks"]} blocks',
                    transform=axes[0, 0].transAxes,
                    ha='right', va='bottom', fontsize=7,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
    axes[0, 0].legend(loc='lower left', fontsize=7)
    grid = np.arange(0, 60001, 250.)
    for arm, result, rows in (('spare', spare_run, spare_rows),
                              ('srp', srp_run, srp_rows)):
        axes[0, 1].plot(grid/1000, waiting_series(result['probe_records'], grid),
                        color=colors[arm], lw=1.4 if arm == 'spare' else 1.,
                        label=arm.upper() if arm == 'srp' else 'SPARE')
        t = np.array([r['simulated_finish_ms']/1000 for r in rows])
        keep = t <= 60
        axes[1, 0].plot(t[keep],
                        [KV-1-r['free_after'] for r, k in zip(rows, keep) if k],
                        color=colors[arm], lw=1.1)
        axes[1, 1].plot(t[keep],
                        np.cumsum([r['preemptions'] for r in rows])[keep],
                        color=colors[arm], lw=1.4)
    axes[0, 1].set_title('(b) Requests awaiting first token', loc='left')
    axes[1, 0].set_title('(c) Occupied KV blocks', loc='left')
    axes[1, 1].set_title('(d) Cumulative preemptions', loc='left')
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axvspan(20, 60, color='#ededed', alpha=.5, zorder=0)
        ax.axvline(left['simulated_start_ms']/1000, color='#555555',
                   lw=.8, ls='--', zorder=1)
        ax.set_xlim(0, 60)
        ax.grid(axis='y', lw=.4, color='#e1e4e6')
        ax.set_xlabel('Simulated time (s)')
    axes[0, 1].set_ylabel('Requests')
    axes[1, 0].set_ylabel('Blocks')
    axes[1, 1].set_ylabel('Preemptions')
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(folder / f'paired_mechanism_trace.{suffix}', dpi=300,
                    facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    sample, spare_archive, srp_archive = select_sample()
    folder = run.OUT / 'analysis' / 'trace'
    if args.plot_only:
        metadata = run.prior.base.read(folder / 'metadata.json')
        assert metadata['sample'] == sample['name']
        traces = {}
        for arm in ('spare', 'srp'):
            result = run.prior.base.read(folder / arm / 'result.json')
            traces[arm] = (result, iterations(folder / arm / 'iterations.jsonl.gz'))
        choice = first_divergence(traces['spare'][1], traces['srp'][1])
        plot(traces['spare'][0], traces['srp'][0], traces['spare'][1],
             traces['srp'][1], choice, folder)
        print(json.dumps({'plot': 'complete', 'sample': sample['name']}))
        return
    run.prior.initialize()
    traces = {}
    for arm, archived in (('spare', spare_archive), ('srp', srp_archive)):
        result = replay(sample, arm, folder / arm)
        assert result['trajectory_sha256'] == archived['trajectory_sha256']
        for metric in ('mean_ttft_ms', 'p95_ttft_ms', 'p99_tbt_ms',
                       'backlog_drift', 'preemptions', 'recomputed_tokens'):
            assert run.prior.base.probe(result)[metric] == archived[metric]
        traces[arm] = (result, iterations(folder / arm / 'iterations.jsonl.gz'))
    choice = first_divergence(traces['spare'][1], traces['srp'][1])
    left, right = choice
    metadata = dict(source=SOURCE, kv_blocks=KV, rate_qps=RATE,
                    sample_selection='median of 50 paired SRP minus SPARE mean TTFT differences',
                    sample=sample['name'], first_divergence_round=left['round'],
                    simulated_start_ms=left['simulated_start_ms'],
                    free_before=left['free_before'],
                    base_growth_blocks=left['baseline_growth_blocks'],
                    spare_growth_blocks=left['actual_growth_blocks'],
                    threshold=left['threshold'],
                    base_tokens=left['baseline'], spare_tokens=left['scheduled'],
                    spare_window=run.prior.base.probe(traces['spare'][0]),
                    srp_window=run.prior.base.probe(traces['srp'][0]))
    run.prior.base.write(folder / 'metadata.json', metadata)
    print(json.dumps({key: metadata[key] for key in
          ('sample', 'first_divergence_round', 'base_growth_blocks',
           'spare_growth_blocks', 'threshold')}))


if __name__ == '__main__':
    main()
