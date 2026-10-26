"""Resume only missing same-order ablation capacity sweeps; reuse SPARE."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

from research.www_capacity_pareto_20261015 import run as base
from research.www_kv_capacity_50wide_20261016 import run as archive

ROOT = base.ROOT
OUT = ROOT / 'artifacts/www_biobjective_ablation_20261016_r1'
OLD = ROOT / 'artifacts/www_kv_capacity_50wide_20261016_r1'
CAPACITIES = (768, 1024, 1536, 2048, 3072, 4096)
RATES = base.RATES
ARMS = {
    'spare': ('arrival', {}),
    'srp': ('srp', {}),
    'no_arrival': ('arrival', {'tau': 0}),
    'constant16': ('arrival', {'mode': 'constant', 'n': 16}),
    'constant32': ('arrival', {'mode': 'constant', 'n': 32}),
    'no_finish_guard': ('arrival', {'protect_finish': False}),
}
LABELS = {'spare': 'SPARE', 'srp': 'SRP (no saving)',
          'no_arrival': 'Without arrival term', 'constant16': 'Fixed threshold 16',
          'constant32': 'Fixed threshold 32', 'no_finish_guard': 'Without first-token protection'}


def write(path, value):
    base.write(path, value)


def prepare():
    old = base.read(OLD / 'design.json')
    samples = old['samples']
    assert all(sum(s['source'] == source for s in samples) == 50
               for source in ('code', 'conversation'))
    assert old['budget'] == 1024 and old['chunk'] == 256
    assert tuple(old['rates']) == RATES and old['reference_qps'] == .5
    assert old['slo'] == {'p95_ttft_ms': 12000., 'p99_tbt_ms': 200.}
    for path, sha in old['input_sha256'].items():
        assert base.digest(ROOT / path) == sha, path
    for path, sha in old['source_sha256'].items():
        assert base.digest(ROOT / path) == sha, path
    assert base.digest(base.PREVIOUS / 'timing_profile.json') == old['timing_sha256']
    sources = (
        'research/www_biobjective_ablation_20261016/run.py',
        'research/www_capacity_pareto_20261015/run.py',
        'research/www_kv_capacity_50wide_20261016/run.py',
        'research/www_evaluation_completion_20261003/simulate.py',
        'research/www_evaluation_completion_20261003/scheduler.py',
        'research/www_evaluation_completion_20261003/policy.py',
        'research/www_evaluation_completion_20261003/timing.py',
        'research/v1_budget_20260922/scheduler.py',
        'research/v1_budget_20260922/policy.py',
        'research/v1_saving_threshold_20260924/policy.py',
        'research/v1_broad12h_20260925/policy.py',
    )
    design = dict(samples=samples, capacities=CAPACITIES, rates=RATES, arms=ARMS,
                  labels=LABELS, budget=1024, chunk=256, concurrency=32,
                  windows_per_source=50, reference_qps=.5, slos=old['slo'],
                  horizon_ms=60000, warm_ms=20000, backlog_interval_ms=[40000, 60000],
                  pass_rule='all 50 valid; mean per-window P95 TTFT <= 12000 ms; '
                  'mean per-window P99 TBT <= 200 ms; mean backlog count drift <= max(2, QPS)',
                  scan_rule='ascending grid, stop at first failure after measuring 0.5 QPS',
                  timing_sha256=old['timing_sha256'], input_sha256=old['input_sha256'],
                  source_sha256={p: base.digest(ROOT / p) for p in sources},
                  archive_design_sha256=base.digest(OLD / 'design.json'),
                  gpu_execution=False, scheduler_overhead_injected=False,
                  caveat='12 s SLO is inherited from exploratory calibration; '
                  'capacity is a finite-window grid estimate, not a stability proof',
                  reuse='SPARE rate and group records after two replay parity checks; '
                  'old ablations use different inputs/timing and are not combined')
    normalized = json.loads(json.dumps(design))
    path = OUT / 'design.json'
    if path.exists():
        assert base.read(path) == normalized, 'Frozen design changed; use a new output directory'
    else:
        write(path, normalized)
    return normalized


def initialize():
    import torch
    torch.set_num_threads(1)
    base.initialize()
    base.MAX_WALL_S = 180


def simulate_one(sample, kv, arm, rate):
    engine, options = ARMS[arm]
    seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
    requests = base.generate(sample, rate)
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

    identity = dict(source=sample['source'], name=sample['name'], day=sample['day'],
                    capacity=kv, arm=arm, budget=1024, rate=rate)
    try:
        with patch.object(base.SIM, 'make_scheduler', make):
            result = base.SIM.simulate(
                requests, arm=engine, capacity=kv, budget=1024, chunk=256,
                policy_options=options, history=base.HISTORY[sample['source']],
                seed=seed, fitted_timing=Clock(), capture_probe=True,
                output=None, max_wall_s=180)
    except RuntimeError as exc:
        if str(exc) not in ('empty schedule with pending work; trace/policy cannot progress',
                            'simulation limit; no completed result will be reported'):
            raise
        return dict(identity, valid=False, failure=str(exc), requests=len(requests))
    assert result['metrics']['completed'] == len(requests)
    return dict(identity, valid=True, **base.probe(result),
                interventions=result['metrics']['interventions'],
                rounds=result['metrics']['rounds'])


def smoke():
    design = prepare()
    initialize()
    checks = []
    for source, rate in (('code', .5), ('conversation', 1.)):
        sample = next(s for s in design['samples'] if s['source'] == source)
        prior = base.read(OLD / 'rates' / f'{source}_c768_arrival_r{rate:g}.json')
        old = next(r for r in prior['rows'] if r['name'] == sample['name'])
        new = simulate_one(sample, 768, 'spare', rate)
        for key in ('valid', 'trajectory_sha256', 'mean_ttft_ms', 'p95_ttft_ms',
                    'p99_tbt_ms', 'backlog_drift', 'preemptions', 'recomputed_tokens'):
            assert new[key] == old[key], (source, key, new[key], old[key])
        checks.append(dict(source=source, rate=rate, name=sample['name'],
                           trajectory_sha256=new['trajectory_sha256'], parity=True))
    sample = design['samples'][0]
    variants = {}
    for arm in ARMS:
        row = simulate_one(sample, 768, arm, 1.)
        assert row['valid'], (arm, row)
        variants[arm] = {k: row[k] for k in ('trajectory_sha256', 'mean_ttft_ms', 'interventions')}
    report = dict(passed=True, design_sha256=base.digest(OUT / 'design.json'),
                  archive_parity=checks, variants=variants)
    write(OUT / 'smoke.json', report)
    print(json.dumps(report, indent=2), flush=True)


def group_path(source, kv, arm):
    return OUT / 'groups' / f'{source}_c{kv}_{arm}.json'


def rate_path(source, kv, arm, rate):
    return OUT / 'rates' / f'{source}_c{kv}_{arm}_r{rate:g}.json'


def get_rate(source, kv, arm, rate, samples):
    path = rate_path(source, kv, arm, rate)
    wanted = {s['name'] for s in samples}
    if path.exists():
        record = base.read(path)
        assert {r['name'] for r in record['rows']} == wanted
        assert record['summary'] == archive.summary(record['rows'], rate)
        return record
    if arm == 'spare':
        prior = OLD / 'rates' / f'{source}_c{kv}_arrival_r{rate:g}.json'
        original = base.read(prior)
        rows = [dict(r, arm='spare') for r in original['rows']]
        assert {r['name'] for r in rows} == wanted
        assert archive.summary(rows, rate) == original['summary']
        record = dict(rows=rows, summary=original['summary'], reused_rows=len(rows),
                      newly_simulated=0, reused_from=str(prior.relative_to(ROOT)),
                      reused_sha256=base.digest(prior))
    else:
        checkpoint = OUT / 'checkpoints' / path.name
        rows = base.read(checkpoint)['rows'] if checkpoint.exists() else []
        found = {r['name'] for r in rows}
        assert len(found) == len(rows) and found <= wanted
        for sample in samples:
            if sample['name'] in found:
                continue
            rows.append(simulate_one(sample, kv, arm, rate))
            write(checkpoint, dict(rows=rows, updated_at=time.time()))
            write(OUT / 'workers' / f'{source}_c{kv}_{arm}.json',
                  dict(source=source, kv=kv, arm=arm, rate=rate, rows=len(rows),
                       updated_at=time.time(), pid=os.getpid()))
        rows.sort(key=lambda r: r['name'])
        record = dict(rows=rows, summary=archive.summary(rows, rate),
                      reused_rows=0, newly_simulated=len(rows))
    write(path, record)
    return record


def run_group(task):
    source, kv, arm, samples = task
    target = group_path(source, kv, arm)
    if target.exists():
        return base.read(target)
    capacity, failed = 0., None
    tested = []
    for rate in RATES:
        record = get_rate(source, kv, arm, rate, samples)
        tested.append(rate)
        if failed is None:
            if record['summary']['passed']:
                capacity = rate
            else:
                failed = rate
        if failed is not None and rate >= .5:
            break
    result = dict(source=source, kv_blocks=kv, arm=arm, capacity_qps=capacity,
                  next_failed_qps=failed, capacity_is_lower_bound=failed is None,
                  reference_feasible=capacity >= .5, tested_rates=tested,
                  windows=50, updated_at=time.time())
    write(target, result)
    return result


def progress():
    paths = list((OUT / 'rates').glob('*.json'))
    new = reused = failed = partial = 0
    for path in paths:
        record = base.read(path)
        new += record['newly_simulated']
        reused += record['reused_rows']
        failed += record['summary']['failed_samples']
    for path in (OUT / 'checkpoints').glob('*.json'):
        if not (OUT / 'rates' / path.name).exists():
            partial += len(base.read(path)['rows'])
    return dict(groups_complete=len(list((OUT / 'groups').glob('*.json'))),
                groups_total=2 * len(CAPACITIES) * len(ARMS), rates_complete=len(paths),
                new_rows=new + partial, reused_rows=reused, invalid_rows=failed,
                updated_at=time.time())


def run(workers):
    design = prepare()
    proof = base.read(OUT / 'smoke.json')
    assert proof['passed'] and proof['design_sha256'] == base.digest(OUT / 'design.json')
    lock = (OUT / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(OUT / 'pid.json', dict(pid=os.getpid(), host=os.uname().nodename))
    os.nice(5)
    tasks = [(source, kv, arm, [s for s in design['samples'] if s['source'] == source])
             for kv in CAPACITIES for source in ('code', 'conversation') for arm in ARMS]
    started = time.monotonic()
    errors = []
    try:
        # Copy complete, proven-compatible SPARE points without starting new simulations.
        for task in tasks:
            if task[2] == 'spare':
                run_group(task)
        pending = [t for t in tasks if not group_path(*t[:3]).exists()]
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
            futures = {pool.submit(run_group, t): t[:3] for t in pending}
            while futures:
                ready, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
                for future in ready:
                    name = futures.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append(dict(group=name, error=repr(exc)))
                status = dict(stage='running', pid=os.getpid(), workers=workers,
                              elapsed_s=round(time.monotonic()-started, 1),
                              errors=errors, **progress())
                write(OUT / 'status.json', status)
                print(json.dumps(status), flush=True)
        if errors:
            raise RuntimeError(f'{len(errors)} groups failed; see status.json')
        assert progress()['groups_complete'] == len(tasks)
        write(OUT / 'status.json', dict(stage='analyzing', errors=[], **progress()))
        subprocess.run([str(ROOT / 'simulator/bin/python'), '-B', '-m',
                        'research.www_biobjective_ablation_20261016.analyze'],
                       cwd=ROOT, check=True)
        write(OUT / 'status.json', dict(stage='complete', errors=[], **progress()))
    except Exception as exc:
        write(OUT / 'status.json', dict(stage='failed', errors=errors, fatal=repr(exc), **progress()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'smoke', 'run', 'status', 'analyze'))
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps({'samples': len(prepare()['samples']), 'groups': 72}))
    elif args.command == 'smoke':
        smoke()
    elif args.command == 'run':
        run(args.workers)
    elif args.command == 'status':
        result = base.read(OUT / 'status.json') if (OUT / 'status.json').exists() else {}
        result.update(progress())
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        subprocess.run([str(ROOT / 'simulator/bin/python'), '-B', '-m',
                        'research.www_biobjective_ablation_20261016.analyze'],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
