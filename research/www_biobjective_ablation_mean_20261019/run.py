"""Paired capacity scans for SPARE component ablations on frozen test windows."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

from research.www_biobjective_ablation_20261016 import run as prior
from research.www_mean_ttft_capacity_20261017 import run as main

ROOT = prior.ROOT
OUT = ROOT / 'artifacts/www_biobjective_ablation_mean_20261019_r2'
KVS = (1024, 2048, 4096)
THETA_MS = (1000., 2000., 4000., 8000.)
SOURCES = ('code', 'conversation')
ARMS = {
    'spare': ('arrival', {}),
    'srp': ('srp', {}),
    'no_finish_guard': ('arrival', {'protect_finish': False}),
    'no_decode_guard': ('arrival', {'protect_decode': False}),
    'no_protection': ('arrival', {'protect_finish': False, 'protect_decode': False}),
    'constant16': ('arrival', {'mode': 'constant', 'n': 16}),
    'constant32': ('arrival', {'mode': 'constant', 'n': 32}),
    'no_arrival': ('arrival', {'tau': 0}),
    'budget_only': ('arrival', {'candidate_axes': 'budget'}),
    'chunk_only': ('arrival', {'candidate_axes': 'chunk'}),
    'token_proxy': ('arrival', {'saving_basis': 'tokens'}),
    'fixed256': ('arrival', {'mode': 'fixed', 'fixed_budget': 256}),
    'fixed512': ('arrival', {'mode': 'fixed', 'fixed_budget': 512}),
}
OLD_ARMS = {
    'spare': ((main.OUT, 'arrival'), (prior.OUT, 'spare')),
    'srp': ((prior.OUT, 'srp'),),
    'no_finish_guard': ((prior.OUT, 'no_finish_guard'),),
    'constant16': ((prior.OUT, 'constant16'),),
    'constant32': ((prior.OUT, 'constant32'),),
    'no_arrival': ((prior.OUT, 'no_arrival'),),
    'no_decode_guard': ((ROOT / 'artifacts/www_biobjective_mechanism_20261016_r1', 'no_decode_guard'),),
    'budget_only': ((ROOT / 'artifacts/www_biobjective_mechanism_20261016_r1', 'budget_only'),),
    'chunk_only': ((ROOT / 'artifacts/www_biobjective_mechanism_20261016_r1', 'chunk_only'),),
    'token_proxy': ((ROOT / 'artifacts/www_biobjective_mechanism_20261016_r1', 'token_proxy'),),
}


def initialize():
    prior.initialize()
    prior.ARMS.update(ARMS)


def prepare():
    frozen = prior.prepare()
    current = prior.base.read(main.OUT / 'design.json')
    assert {(s['name'], s['input']) for s in frozen['samples']} == {
        (s['name'], s['input']) for s in current['samples']}
    assert frozen['timing_sha256'] == current['timing_sha256']
    design = dict(samples=frozen['samples'], kv_blocks=KVS, theta_ms=THETA_MS,
                  sources=SOURCES, arms=ARMS, windows_per_source=50,
                  budget=1024, chunk=256, concurrency=32,
                  horizon_ms=60000, warm_ms=20000,
                  backlog_interval_ms=(40000, 60000),
                  p99_tbt_limit_ms=200,
                  capacity_rule='all 50 windows valid; mean window TTFT <= theta; '
                                'mean window P99 TBT <= 200 ms; mean backlog drift <= max(2,QPS); '
                                'ascending QPS scan through first failure',
                  refinement_qps=.02, simulator_only=True,
                  prior_design_sha256=prior.base.digest(prior.OUT / 'design.json'),
                  main_design_sha256=prior.base.digest(main.OUT / 'design.json'),
                  runner_sha256=prior.base.digest(Path(__file__)),
                  policy_sha256=prior.base.digest(ROOT / 'research/www_evaluation_gpu_completion_20261003/policy.py'),
                  timing_sha256=frozen['timing_sha256'])
    path = OUT / 'design.json'
    if path.exists():
        assert prior.base.read(path) == json.loads(json.dumps(design))
    else:
        prior.base.write(path, design)
    return design


def rate_name(source, kv, arm, rate):
    return f'{source}_c{kv}_{arm}_r{rate:g}.json'


def rate_path(source, kv, arm, rate):
    return OUT / 'rates' / rate_name(source, kv, arm, rate)


def cached_record(source, kv, arm, rate):
    if arm == 'spare':
        path = main.cached_path(source, kv, 'arrival', rate)
        if path is not None:
            return path
    for root, old_arm in OLD_ARMS.get(arm, ()):
        path = root / 'rates' / rate_name(source, kv, old_arm, rate)
        if path.exists():
            return path
    return None


def get_rate(source, kv, arm, rate, samples):
    target = rate_path(source, kv, arm, rate)
    if target.exists():
        data = prior.base.read(target)
    else:
        cached = cached_record(source, kv, arm, rate)
        if cached is not None:
            data = prior.base.read(cached)
            data = dict(rows=data['rows'], reused_from=str(cached.relative_to(ROOT)),
                        reused_sha256=prior.base.digest(cached))
        else:
            rows = [prior.simulate_one(sample, kv, arm, rate) for sample in samples]
            data = dict(rows=rows, newly_simulated=len(rows))
        data['summary'] = prior.archive.summary(data['rows'], rate)
        prior.base.write(target, data)
    assert {r['name'] for r in data['rows']} == {s['name'] for s in samples}
    assert data['summary'] == prior.archive.summary(data['rows'], rate)
    return data['summary']


def resource_ok(s, rate):
    return (s['failed_samples'] == 0 and s['p99_tbt_ms'] <= 200
            and s['backlog_drift'] <= max(2, rate))


def bounds(scans, theta):
    last_pass = 0.
    first_fail = None
    cause = None
    for rate in sorted(scans):
        s = scans[rate]
        reasons = []
        if s['failed_samples']:
            reasons.append('invalid')
        if s['mean_ttft_ms'] is not None and s['mean_ttft_ms'] > theta:
            reasons.append('mean_ttft')
        if s['p99_tbt_ms'] is not None and s['p99_tbt_ms'] > 200:
            reasons.append('p99_tbt')
        if s['backlog_drift'] is not None and s['backlog_drift'] > max(2, rate):
            reasons.append('backlog')
        if reasons:
            first_fail, cause = rate, reasons
            break
        last_pass = rate
    return last_pass, first_fail, cause


def run_group(task):
    source, kv, arm, samples = task
    target = OUT / 'groups' / f'{source}_c{kv}_{arm}.json'
    if target.exists():
        return prior.base.read(target)
    scans = {}
    coarse = sorted(set(main.COARSE_RATES) | {.5})
    first_resource_fail = None
    for rate in coarse:
        scans[rate] = get_rate(source, kv, arm, rate, samples)
        if not resource_ok(scans[rate], rate):
            first_resource_fail = rate
            break
    while first_resource_fail is None:
        rate = 2 * max(scans)
        if rate > 64:
            raise RuntimeError(f'No resource failure: {source}/{kv}/{arm}')
        scans[rate] = get_rate(source, kv, arm, rate, samples)
        if not resource_ok(scans[rate], rate):
            first_resource_fail = rate
    for theta in THETA_MS:
        low, high, _ = bounds(scans, theta)
        assert high is not None
        while high - low > .0200001:
            rate = round((low + high) * 50) / 100
            if not low < rate < high:
                break
            scans[rate] = get_rate(source, kv, arm, rate, samples)
            if resource_ok(scans[rate], rate) and scans[rate]['mean_ttft_ms'] <= theta:
                low = rate
            else:
                high = rate
    capacities = {}
    for theta in THETA_MS:
        low, high, cause = bounds(scans, theta)
        capacities[f'{theta:g}'] = dict(last_pass_qps=low, first_fail_qps=high,
                                        resolution_qps=round(high-low, 4), failure_causes=cause)
    result = dict(source=source, kv_blocks=kv, arm=arm, capacities=capacities,
                  rates=sorted(scans), newly_simulated_rates=sum(
                      'newly_simulated' in prior.base.read(rate_path(source, kv, arm, r)) for r in scans),
                  completed_at=time.time())
    prior.base.write(target, result)
    return result


def smoke():
    d = prepare()
    initialize()
    sample = next(s for s in d['samples'] if s['source'] == 'conversation')
    full = prior.simulate_one(sample, 2048, 'spare', 3.)
    old = prior.base.read(main.cached_path('conversation', 2048, 'arrival', 3.))
    assert full['trajectory_sha256'] == next(r['trajectory_sha256'] for r in old['rows']
                                              if r['name'] == sample['name'])
    no_deferral = prior.simulate_one(sample, 2048, 'srp', 3.)
    prior.ARMS['srp_without_protection'] = ('srp', {'protect_finish': False,
                                                    'protect_decode': False})
    both_off = prior.simulate_one(sample, 2048, 'srp_without_protection', 3.)
    assert no_deferral['trajectory_sha256'] == both_off['trajectory_sha256']
    no_protection = prior.simulate_one(sample, 2048, 'no_protection', 3.)
    result = dict(full_parity=True, no_deferral_equals_no_deferral_no_protection=True,
                  no_protection_valid=no_protection['valid'],
                  sample=sample['name'])
    prior.base.write(OUT / 'smoke.json', result)
    print(json.dumps(result, indent=2))


def run(workers):
    d = prepare()
    assert prior.base.read(OUT / 'smoke.json')['full_parity']
    tasks = [(source, kv, arm, [s for s in d['samples'] if s['source'] == source])
             for kv in KVS for source in SOURCES for arm in ARMS]
    started = time.monotonic()
    done, errors = 0, []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        futures = {pool.submit(run_group, task): task[:3] for task in tasks}
        for future in as_completed(futures):
            try:
                future.result()
                done += 1
            except Exception as exc:
                errors.append(dict(task=futures[future], error=repr(exc)))
            if done % 6 == 0 or errors:
                status = dict(completed=done, total=len(tasks), errors=errors,
                              elapsed_s=round(time.monotonic()-started, 1))
                prior.base.write(OUT / 'status.json', status)
                print(status, flush=True)
    status = dict(completed=done, total=len(tasks), errors=errors,
                  elapsed_s=round(time.monotonic()-started, 1))
    prior.base.write(OUT / 'status.json', status)
    if errors:
        raise RuntimeError(status)
    print(status, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    smoke() if args.smoke else run(args.workers)
