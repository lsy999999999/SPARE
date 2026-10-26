"""Resume missing component sweeps on the frozen 50-window capacity streams."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

from research.www_biobjective_ablation_20261016 import run as prior
from research.www_fixed512_capacity_20261014.policy import select as extended_select

ROOT = prior.ROOT
OUT = ROOT / 'artifacts/www_biobjective_mechanism_20261016_r1'
CAPACITIES = (1024, 1536, 2048, 3072, 4096)
OPTIONS = {
    'budget_only': {'candidate_axes': 'budget'},
    'chunk_only': {'candidate_axes': 'chunk'},
    'token_proxy': {'saving_basis': 'tokens'},
    'no_decode_guard': {'protect_decode': False},
}


def save(path, value):
    prior.base.write(path, value)


def prepare():
    frozen = prior.prepare()
    source_paths = (
        'research/www_biobjective_mechanism_20261016/run.py',
        'research/www_fixed512_capacity_20261014/policy.py',
    )
    design = dict(source_design_sha256=prior.base.digest(prior.OUT / 'design.json'),
                  source_sha256={p: prior.base.digest(ROOT / p) for p in source_paths},
                  capacities=CAPACITIES, options=OPTIONS,
                  sources=('code', 'conversation'), windows_per_source=50,
                  rates=prior.RATES, budget=1024, chunk=256, concurrency=32,
                  reference_qps=.5, slo=frozen['slos'],
                  horizon_ms=60000, warm_ms=20000,
                  backlog_interval_ms=(40000, 60000),
                  timing_sha256=frozen['timing_sha256'],
                  capacity_rule=frozen['pass_rule'],
                  note='CPU fitted clock; no scheduler overhead injected')
    target = OUT / 'design.json'
    if target.exists():
        assert prior.base.read(target) == json.loads(json.dumps(design))
    else:
        save(target, design)
    return frozen


@contextmanager
def extended_policy():
    from research.www_evaluation_completion_20261003 import scheduler
    with patch.object(scheduler, 'select', extended_select):
        yield


def simulate(sample, kv, arm, rate):
    original = prior.ARMS.get(arm)
    prior.ARMS[arm] = ('arrival', OPTIONS.get(arm, {}))
    try:
        with extended_policy():
            return prior.simulate_one(sample, kv, arm, rate)
    finally:
        if original is None:
            del prior.ARMS[arm]
        else:
            prior.ARMS[arm] = original


def smoke():
    frozen = prepare()
    prior.initialize()
    samples = frozen['samples']
    checks = []
    for source, kv, rate in (('code', 1024, 1.), ('conversation', 2048, 3.),
                             ('conversation', 4096, 4.)):
        sample = next(s for s in samples if s['source'] == source)
        reference = prior.simulate_one(sample, kv, 'spare', rate)
        patched = simulate(sample, kv, 'default_parity', rate)
        assert reference['trajectory_sha256'] == patched['trajectory_sha256']
        for metric in ('mean_ttft_ms', 'p95_ttft_ms', 'p99_tbt_ms',
                       'backlog_drift', 'preemptions', 'recomputed_tokens'):
            assert reference[metric] == patched[metric], (source, kv, metric)
        checks.append(dict(source=source, kv=kv, rate=rate,
                           trajectory_sha256=reference['trajectory_sha256']))
    variants = {}
    sample = next(s for s in samples if s['source'] == 'conversation')
    for arm in OPTIONS:
        row = simulate(sample, 2048, arm, 3.)
        assert row['valid'], (arm, row)
        variants[arm] = dict(trajectory_sha256=row['trajectory_sha256'],
                             interventions=row['interventions'])
    save(OUT / 'smoke.json', dict(passed=True, checks=checks, variants=variants))
    print(json.dumps(dict(passed=True, checks=len(checks), variants=variants)))


def run_group(task):
    source, kv, arm, samples = task
    group_path = OUT / 'groups' / f'{source}_c{kv}_{arm}.json'
    if group_path.exists():
        return prior.base.read(group_path)
    capacity, failure, tested = 0., None, []
    for rate in prior.RATES:
        path = OUT / 'rates' / f'{source}_c{kv}_{arm}_r{rate:g}.json'
        if path.exists():
            record = prior.base.read(path)
            assert {r['name'] for r in record['rows']} == {s['name'] for s in samples}
            assert record['summary'] == prior.archive.summary(record['rows'], rate)
        else:
            checkpoint = OUT / 'checkpoints' / path.name
            rows = prior.base.read(checkpoint)['rows'] if checkpoint.exists() else []
            found = {r['name'] for r in rows}
            assert len(found) == len(rows)
            for sample in samples:
                if sample['name'] not in found:
                    rows.append(simulate(sample, kv, arm, rate))
                    save(checkpoint, dict(rows=rows, updated_at=time.time()))
                    found.add(sample['name'])
            rows.sort(key=lambda r: r['name'])
            record = dict(rows=rows, summary=prior.archive.summary(rows, rate))
            save(path, record)
        tested.append(rate)
        if failure is None:
            if record['summary']['passed']:
                capacity = rate
            else:
                failure = rate
        if failure is not None and rate >= .5:
            break
    group = dict(source=source, kv_blocks=kv, arm=arm,
                 capacity_qps=capacity, next_failed_qps=failure,
                 capacity_is_lower_bound=failure is None,
                 tested_rates=tested, windows=50)
    save(group_path, group)
    return group


def run(workers):
    frozen = prepare()
    assert prior.base.read(OUT / 'smoke.json')['passed']
    tasks = [(source, kv, arm, [s for s in frozen['samples'] if s['source'] == source])
             for source in ('code', 'conversation') for kv in CAPACITIES for arm in OPTIONS]
    pending = [t for t in tasks if not (OUT / 'groups' /
               f'{t[0]}_c{t[1]}_{t[2]}.json').exists()]
    errors = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=prior.initialize) as pool:
        futures = {pool.submit(run_group, task): task[:3] for task in pending}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append(dict(group=futures[future], error=repr(exc)))
            status = dict(stage='running', complete=len(tasks)-len(pending)+
                          sum((OUT / 'groups' / f'{t[0]}_c{t[1]}_{t[2]}.json').exists()
                              for t in pending), total=len(tasks), errors=errors,
                          elapsed_s=round(time.monotonic()-started, 1))
            save(OUT / 'status.json', status)
            print(json.dumps(status), flush=True)
    status['stage'] = 'complete' if not errors and status['complete'] == len(tasks) else 'failed'
    save(OUT / 'status.json', status)
    if status['stage'] != 'complete':
        raise RuntimeError(status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'smoke', 'run'))
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(dict(samples=len(prepare()['samples']), groups=40)))
    elif args.command == 'smoke':
        smoke()
    else:
        run(args.workers)


if __name__ == '__main__':
    main()
