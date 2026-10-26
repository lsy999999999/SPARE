"""Scan mean-TTFT constrained capacity for paired component ablations."""
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

from research.www_biobjective_ablation_mean_20261019 import run as old
from research.www_component_ablation_mean_20261019 import policy

ROOT = old.ROOT
OUT = ROOT / 'artifacts/www_component_ablation_mean_20261019_r2'
PREVIOUS = ROOT / 'artifacts/www_component_ablation_mean_20261019_r1'
WINDOWS = (2, 6, 10, 14, 18, 22)
KVS = old.KVS
THETA_MS = old.THETA_MS
SOURCES = old.SOURCES
NEW = {
    'constant1': ('arrival', {'mode': 'constant', 'n': 1}),
    'constant8': ('arrival', {'mode': 'constant', 'n': 8}),
    'chunk_family': ('arrival', {'candidate_indices': (3, 4)}),
    'drop_candidate1': ('arrival', {'candidate_indices': (2, 3, 4)}),
    'drop_candidate2': ('arrival', {'candidate_indices': (1, 3, 4)}),
    'drop_candidate3': ('arrival', {'candidate_indices': (1, 2, 4)}),
    'drop_candidate4': ('arrival', {'candidate_indices': (1, 2, 3)}),
}
EXISTING = ('spare', 'srp', 'no_finish_guard', 'no_decode_guard',
            'no_protection', 'no_arrival', 'constant16', 'constant32',
            'budget_only', 'chunk_only', 'token_proxy')
ARMS = EXISTING + tuple(NEW)


def initialize():
    old.initialize()
    old.prior.ARMS.update(NEW)
    from research.www_evaluation_completion_20261003 import scheduler
    scheduler.select = policy.select


def prepare():
    prior = old.prepare()
    samples = [s for s in prior['samples'] if s['window'] in WINDOWS]
    assert len(samples) == 24
    assert all(sum(s['source'] == source for s in samples) == 12 for source in SOURCES)
    design = dict(samples=samples, windows=WINDOWS, sources=SOURCES,
                  kv_blocks=KVS, theta_ms=THETA_MS, arms=ARMS,
                  existing=EXISTING, new=NEW, budget=1024, chunk=256,
                  concurrency=32, horizon_ms=60000, warm_ms=20000,
                  backlog_interval_ms=(40000, 60000), p99_tbt_limit_ms=200,
                  capacity_rule='all 12 windows valid; mean TTFT <= theta; '
                                'mean window P99 TBT <= 200 ms; mean backlog drift '
                                '<= max(2,QPS); ascending scan through first failure',
                  refinement_qps=.02, gpu_execution=False,
                  prior_design_sha256=old.prior.base.digest(old.OUT / 'design.json'),
                  old_policy_sha256=old.prior.base.digest(
                      ROOT / 'research/www_evaluation_gpu_completion_20261003/policy.py'),
                  policy_sha256=old.prior.base.digest(Path(policy.__file__)),
                  runner_sha256=old.prior.base.digest(Path(__file__)))
    path = OUT / 'design.json'
    if path.exists():
        assert old.prior.base.read(path) == json.loads(json.dumps(design))
    else:
        old.prior.base.write(path, design)
    return samples


def rate_path(source, kv, arm, rate):
    return OUT / 'rates' / old.rate_name(source, kv, arm, rate)


def summary(rows, rate):
    assert len(rows) == 12 and len({r['name'] for r in rows}) == 12
    valid = [r for r in rows if r['valid']]
    result = {key: sum(r[key] for r in valid)/len(valid) if valid else None
              for key in ('mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift',
                          'preemptions', 'recomputed_tokens')}
    result['failed_samples'] = len(rows)-len(valid)
    result['drift_limit'] = max(2, rate)
    return result


def get_rate(source, kv, arm, rate, samples):
    target = rate_path(source, kv, arm, rate)
    wanted = {s['name'] for s in samples}
    if target.exists():
        data = old.prior.base.read(target)
    else:
        cached = PREVIOUS / 'rates' / old.rate_name(source, kv, arm, rate)
        if not cached.exists():
            cached = None
        if arm in EXISTING:
            path = old.rate_path(source, kv, arm, rate)
            cached = cached or (path if path.exists() else old.cached_record(source, kv, arm, rate))
        if cached is not None:
            original = old.prior.base.read(cached)
            rows = [r for r in original['rows'] if r['name'] in wanted]
            data = dict(rows=rows, reused_from=str(cached.relative_to(ROOT)),
                        reused_sha256=old.prior.base.digest(cached))
        else:
            rows = [old.prior.simulate_one(sample, kv, arm, rate) for sample in samples]
            data = dict(rows=rows, newly_simulated=len(rows))
        data['summary'] = summary(data['rows'], rate)
        old.prior.base.write(target, data)
    assert {r['name'] for r in data['rows']} == wanted
    assert data['summary'] == summary(data['rows'], rate)
    return data['summary']


def run_group(task):
    source, kv, arm, samples = task
    target = OUT / 'groups' / f'{source}_c{kv}_{arm}.json'
    if target.exists():
        return old.prior.base.read(target)
    scans = {}
    for rate in sorted(set(old.main.COARSE_RATES) | {.5}):
        scans[rate] = get_rate(source, kv, arm, rate, samples)
        if not old.resource_ok(scans[rate], rate):
            break
    while old.resource_ok(scans[max(scans)], max(scans)):
        rate = 2 * max(scans)
        if rate > 64:
            raise RuntimeError(f'No resource failure: {source}/{kv}/{arm}')
        scans[rate] = get_rate(source, kv, arm, rate, samples)
    for theta in THETA_MS:
        low, high, _ = old.bounds(scans, theta)
        assert high is not None
        while high-low > .0200001:
            rate = round((low+high)*50)/100
            if not low < rate < high:
                break
            scans[rate] = get_rate(source, kv, arm, rate, samples)
            if old.resource_ok(scans[rate], rate) and scans[rate]['mean_ttft_ms'] <= theta:
                low = rate
            else:
                high = rate
    capacities = {}
    for theta in THETA_MS:
        low, high, cause = old.bounds(scans, theta)
        capacities[f'{theta:g}'] = dict(last_pass_qps=low, first_fail_qps=high,
                                        resolution_qps=round(high-low, 4),
                                        failure_causes=cause)
    result = dict(source=source, kv_blocks=kv, arm=arm, capacities=capacities,
                  rates=sorted(scans), completed_at=time.time())
    old.prior.base.write(target, result)
    return result


def smoke():
    samples = prepare()
    initialize()
    checks = []
    for source, kv, rate in (('code', 1024, 2.), ('conversation', 2048, 3.),
                             ('conversation', 4096, 4.)):
        sample = next(s for s in samples if s['source'] == source)
        archived = old.prior.base.read(old.rate_path(source, kv, 'spare', rate))
        reference = next(r for r in archived['rows'] if r['name'] == sample['name'])
        current = old.prior.simulate_one(sample, kv, 'spare', rate)
        assert current['trajectory_sha256'] == reference['trajectory_sha256']
        checks.append(dict(source=source, kv=kv, rate=rate,
                           trajectory_sha256=current['trajectory_sha256']))
    sample = next(s for s in samples if s['source'] == 'conversation')
    variants = {}
    for arm in NEW:
        row = old.prior.simulate_one(sample, 2048, arm, 3.)
        assert row['valid'], (arm, row)
        variants[arm] = dict(trajectory_sha256=row['trajectory_sha256'],
                             interventions=row['interventions'])
    old.prior.base.write(OUT / 'smoke.json', dict(parity=checks, variants=variants))
    print(json.dumps(dict(parity=len(checks), variants=variants), indent=2))


def run(workers):
    samples = prepare()
    assert old.prior.base.read(OUT / 'smoke.json')['parity']
    tasks = [(source, kv, arm, [s for s in samples if s['source'] == source])
             for source in SOURCES for kv in KVS for arm in ARMS]
    pending = [task for task in tasks if not (OUT / 'groups' /
               f'{task[0]}_c{task[1]}_{task[2]}.json').exists()]
    done, errors = len(tasks)-len(pending), []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        futures = {pool.submit(run_group, task): task[:3] for task in pending}
        for future in as_completed(futures):
            try:
                future.result()
                done += 1
            except Exception as exc:
                errors.append(dict(task=futures[future], error=repr(exc)))
            if done % 6 == 0 or errors:
                status = dict(completed=done, total=len(tasks), errors=errors,
                              elapsed_s=round(time.monotonic()-started, 1))
                old.prior.base.write(OUT / 'status.json', status)
                print(status, flush=True)
    status = dict(completed=done, total=len(tasks), errors=errors,
                  elapsed_s=round(time.monotonic()-started, 1))
    old.prior.base.write(OUT / 'status.json', status)
    if errors:
        raise RuntimeError(status)
    print(status, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    smoke() if args.smoke else run(args.workers)
