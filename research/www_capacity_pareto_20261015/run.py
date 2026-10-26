"""Measure steady-load request capacity on paired CPU-simulator streams."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.www_pareto_20261014.run import ROOT, digest, read, write
from research.www_pareto_papers_20261014.run import (
    DIAGNOSTIC, ENHANCED, LEGACY, MAIN, SCOPES,
)

OUT = ROOT / 'artifacts/www_capacity_pareto_20261015_r1'
PREVIOUS = ROOT / 'artifacts/www_pareto_20261014_r1'
CAPACITIES = (768, 4096)
BUDGETS = (256, 512, 1024, 2048)
RATES = (.25, .5, .75, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 8., 10.)
HORIZON_MS = 60000.
WARM_MS = 20000.
MID_MS = 40000.
MAX_WALL_S = 120
SLOS = {'strict': {'p95_ttft_ms': 5000., 'p99_tbt_ms': 200.},
        'relaxed': {'p95_ttft_ms': 10000., 'p99_tbt_ms': 500.}}


def generate(sample, rate):
    pool = read(ROOT / sample['input'])
    seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:8], 'big')
    arrival_rng = np.random.default_rng(seed)
    length_rng = np.random.default_rng(seed ^ 0x9E3779B97F4A7C15)
    gaps = arrival_rng.exponential(1000., 4096)
    times = np.cumsum(gaps) / rate
    if times[-1] < HORIZON_MS:
        raise RuntimeError('arrival stream too short')
    count = int(np.searchsorted(times, HORIZON_MS))
    picks = length_rng.integers(0, len(pool), size=count)
    return [dict(id=f"{sample['name']}:{i}", prompt=pool[int(j)]['prompt'],
                 output=pool[int(j)]['output'], arrival_ms=float(times[i]))
            for i, j in enumerate(picks)]


def initialize():
    global SIM, TIMING, HISTORY, COST, ORIGINAL_MAKE, LEGACY_CLASS, ENHANCED_CLASS
    from research.www_evaluation_completion_20261003 import simulate as sim
    from research.www_evaluation_completion_20261003.timing import FittedTiming
    from research.www_legacy_baselines_20261006.scheduler import GPUBaselineScheduler as Legacy
    from research.www_baseline_fidelity_20261013.scheduler import GPUBaselineScheduler as Enhanced
    SIM, ORIGINAL_MAKE = sim, sim.make_scheduler
    LEGACY_CLASS, ENHANCED_CLASS = Legacy, Enhanced
    sim.digest = lru_cache(maxsize=None)(sim.digest)
    TIMING = FittedTiming(read(PREVIOUS / 'timing_profile.json'))
    HISTORY = {key: [r['output'] for r in rows]
               for key, rows in read(PREVIOUS / 'history.json').items()}
    COST = read(ROOT / 'artifacts/www_baseline_fidelity_20261013_r1/calibration/cost_model.json')


def simulate(requests, arm, capacity, budget, source, seed):
    capture = {}

    def make(arm='arrival', blocks=768, budget=1024, chunk=256, **kwargs):
        if arm in LEGACY + ENHANCED:
            cls = LEGACY_CLASS if arm in LEGACY else ENHANCED_CLASS
            with patch.object(SIM, 'BudgetScheduler', cls):
                scheduler, clock = ORIGINAL_MAKE('native', blocks, budget, chunk, **kwargs)
            scheduler.history_context = kwargs.get('history', ())
            scheduler.seed_context = kwargs.get('seed', 0)
            scheduler.oracle_lengths_context = kwargs.get('oracle_lengths') or {}
            scheduler.fidelity_cost_model = COST
            scheduler.configure(arm, budget, chunk, kwargs.get('concurrency', 32))
        else:
            scheduler, clock = ORIGINAL_MAKE(arm, blocks, budget, chunk, **kwargs)
        capture['scheduler'] = scheduler
        return scheduler, clock

    class Clock:
        def estimate(self, batch):
            duration, details = TIMING.estimate(batch)
            capture['scheduler'].round_rows[-1]['step_ms'] = duration
            return duration, details

    with patch.object(SIM, 'make_scheduler', make), \
         patch.object(SIM, 'ARMS', tuple(set(SIM.ARMS) | set(MAIN))), \
         patch.object(SIM, 'ADAPTATION', dict(SIM.ADAPTATION, **{
             a: ('paper_adapter', (SCOPES[a],)) for a in MAIN})):
        return SIM.simulate(requests, arm=arm, capacity=capacity, budget=budget, chunk=256,
                            history=HISTORY[source], seed=seed, fitted_timing=Clock(),
                            capture_probe=True, output=None, max_wall_s=MAX_WALL_S)


def probe(result, warm=WARM_MS, mid=MID_MS, horizon=HORIZON_MS):
    records = result['probe_records']
    observed = [r for r in records if warm <= r['arrival_ms'] < horizon]
    gaps = [gap for r in observed for gap in result['probe_intervals_ms'][r['request_id']]]
    if not observed:
        raise RuntimeError('no measured requests')

    def outstanding(at):
        return sum(r['arrival_ms'] <= at < r['finish_ms'] for r in records)

    return dict(measured_requests=len(observed),
                mean_ttft_ms=float(np.mean([r['ttft_ms'] for r in observed])),
                p95_ttft_ms=float(np.percentile([r['ttft_ms'] for r in observed], 95)),
                p99_tbt_ms=float(np.percentile(gaps, 99)) if gaps else 0.,
                pending_mid=outstanding(mid), pending_end=outstanding(horizon),
                backlog_drift=outstanding(horizon)-outstanding(mid),
                completed_by_end=sum(r['finish_ms'] <= horizon for r in records),
                arrivals_by_end=len(records),
                preemptions=result['metrics']['preemptions'],
                recomputed_tokens=result['metrics']['recomputed_tokens'],
                trajectory_sha256=result['trajectory_sha256'])


def design():
    source = read(PREVIOUS / 'design.json')
    samples = source['samples']
    for sample in samples:
        assert digest(ROOT / sample['input']) == source['hashes'][sample['input']]
    return dict(samples=samples, source_design=str(PREVIOUS / 'design.json'),
                source_design_sha256=digest(PREVIOUS / 'design.json'),
                source_inputs={s['input']: source['hashes'][s['input']] for s in samples},
                timing_sha256=digest(PREVIOUS / 'timing_profile.json'),
                runner_sha256=digest(ROOT / 'research/www_capacity_pareto_20261015/run.py'),
                simulator_sha256=digest(ROOT / 'research/www_evaluation_completion_20261003/simulate.py'),
                main_arms=MAIN, diagnostic_arms=DIAGNOSTIC,
                budgets=BUDGETS, capacities=CAPACITIES, rates=RATES,
                max_wall_s=MAX_WALL_S,
                slos=SLOS, horizon_ms=HORIZON_MS, warm_ms=WARM_MS, mid_ms=MID_MS,
                arrival_process='Poisson, paired rate-scaled exponentials per source window',
                request_lengths='sampled with replacement from each frozen 64-request window',
                validation_windows=20, test_windows=50,
                performance_validated=False, gpu_execution=False)


def task_run(task):
    sample, capacity, arm, budget, rates, split = task
    target = OUT / split / f"{sample['name']}_c{capacity}_{arm}_t{budget}.json"
    rows = read(target) if target.exists() else []
    done = {r['rate'] for r in rows}
    if len(rows) == len(rates) and done == set(rates):
        return len(rows)
    seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
    for rate in rates:
        if rate in done:
            continue
        requests = generate(sample, rate)
        try:
            result = simulate(requests, arm, capacity, budget, sample['source'], seed)
        except RuntimeError as exc:
            if str(exc) != 'simulation limit; no completed result will be reported':
                raise
            data = dict(valid=False, failure='simulation_limit',
                        measured_requests=0, arrivals_by_end=len(requests))
        else:
            data = dict(valid=True, **probe(result))
            assert result['metrics']['completed'] == len(requests)
        rows.append(dict(source=sample['source'], name=sample['name'], day=sample['day'],
                         capacity=capacity, arm=arm, budget=budget, rate=rate, **data))
        rows.sort(key=lambda r: r['rate'])
        write(target, rows)
        done.add(rate)
    return len(rows)


def run(split, jobs, rates):
    frozen = OUT / 'design.json'
    if not frozen.exists():
        write(frozen, design())
    d = read(frozen)
    if d.get('max_wall_s') != MAX_WALL_S or d['runner_sha256'] != digest(ROOT / 'research/www_capacity_pareto_20261015/run.py'):
        d['max_wall_s'] = MAX_WALL_S
        d['runner_sha256'] = digest(ROOT / 'research/www_capacity_pareto_20261015/run.py')
        write(frozen, d)
    for rel, sha in d['source_inputs'].items():
        if digest(ROOT / rel) != sha:
            raise RuntimeError(f'input changed: {rel}')
    if digest(PREVIOUS / 'timing_profile.json') != d['timing_sha256']:
        raise RuntimeError('timing model changed')
    samples = [s for s in d['samples'] if s['split'] == split]
    if split == 'test':
        read(OUT / 'selection.json')
    configs = [(arm, budget) for arm in MAIN for budget in BUDGETS]
    tasks = [(s, cap, arm, budget, rates, split)
             for s in samples for cap in CAPACITIES
             for arm, budget in configs]
    completed = 0
    failures = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=jobs, initializer=initialize) as pool:
        futures = {pool.submit(task_run, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                completed += future.result()
            except Exception as exc:
                sample, cap, arm, budget, _, _ = futures[future]
                failures.append(dict(name=sample['name'], capacity=cap,
                                     arm=arm, budget=budget, error=repr(exc)))
            if completed % 100 == 0 or failures:
                status = dict(completed=completed, expected=len(tasks)*len(rates),
                              failures=failures, elapsed_s=time.monotonic()-start)
                write(OUT / f'{split}_status.json', status)
                print(json.dumps(status), flush=True)
    status = dict(completed=completed, expected=len(tasks)*len(rates),
                  failures=failures, elapsed_s=time.monotonic()-start)
    write(OUT / f'{split}_status.json', status)
    if failures or completed != status['expected']:
        raise RuntimeError(status)


def pilot():
    initialize()
    d = design()
    for source in ('code', 'conversation'):
        sample = next(s for s in d['samples'] if s['source'] == source and s['split'] == 'validation')
        for capacity in CAPACITIES:
            for arm in ('arrival', 'native', 's3_proxy', 'apt_serve_kv_rule'):
                for rate in (2., 5., 8.):
                    requests = generate(sample, rate)
                    result = simulate(requests, arm, capacity, 512, source, 20261015)
                    p = probe(result)
                    print(json.dumps(dict(source=source, capacity=capacity, arm=arm,
                                          rate=rate, requests=len(requests),
                                          mean_ttft_ms=p['mean_ttft_ms'],
                                          p95_ttft_ms=p['p95_ttft_ms'],
                                          p99_tbt_ms=p['p99_tbt_ms'],
                                          drift=p['backlog_drift'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('pilot', 'validation', 'test'))
    parser.add_argument('--jobs', type=int, default=16)
    parser.add_argument('--rates', default=','.join(map(str, RATES)))
    args = parser.parse_args()
    if args.stage == 'pilot':
        pilot()
    else:
        run(args.stage, args.jobs, tuple(float(x) for x in args.rates.split(',')))
