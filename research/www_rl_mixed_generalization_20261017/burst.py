"""Paired capacity scans under periodic arrival bursts at fixed mean QPS."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import time
from unittest.mock import patch

from research.www_rl_mixed_generalization_20261017 import run, sweep


OUT = run.OUT / 'burst_arrival_20261018'
KV = 2048
PERIOD_MS = 10_000
COARSE = (.25, .5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10., 12.)
TARGET_WIDTH = .02


def burst_requests(sample, rate, duty):
    steady = run.old.base.generate(sample, rate)
    transformed = []
    for request in steady:
        old_time = request['arrival_ms']
        new_time = (old_time // PERIOD_MS) * PERIOD_MS + (old_time % PERIOD_MS) * duty
        transformed.append(dict(request, arrival_ms=new_time))
    assert len(steady) == len(transformed)
    assert all(a['id'] == b['id'] and a['prompt'] == b['prompt']
               and a['output'] == b['output'] for a, b in zip(steady, transformed))
    assert all(a['arrival_ms'] <= b['arrival_ms'] for a, b in
               zip(transformed, transformed[1:]))
    assert all(0 <= r['arrival_ms'] < 60_000 for r in transformed)
    return transformed


def simulate_one(sample, rate, duty, arm, weights):
    requests = burst_requests(sample, rate, duty)
    if arm == 'rl_mixed':
        with patch.object(run.old.base, 'generate', return_value=requests):
            try:
                return run.eval_one(sample, KV, rate, weights)
            except RuntimeError as exc:
                if str(exc) != 'no measured requests':
                    raise
                failure = str(exc)
    else:
        if not hasattr(run.old.base, 'SIM'):
            run.old.base.initialize()
        seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
        try:
            result = run.old.base.simulate(
                requests, 'arrival', KV, 1024, sample['source'], seed)
            return dict(source=sample['source'], name=sample['name'], day=sample['day'],
                        capacity=KV, arm='spare', budget=1024, rate=rate,
                        valid=True, **run.old.base.probe(result),
                        schedule_mean_ms=result['metrics']['cpu_schedule_including_audit_ms_mean'])
        except RuntimeError as exc:
            if str(exc) not in ('no measured requests',
                                'empty schedule with pending work; trace/policy cannot progress',
                                'simulation limit; no completed result will be reported'):
                raise
            failure = str(exc)
    return dict(source=sample['source'], name=sample['name'], day=sample['day'],
                capacity=KV, arm=arm, budget=1024, rate=rate,
                valid=False, failure=failure, requests=len(requests))


def rate_path(source, duty, arm, seed, rate):
    return (OUT / 'rates' /
            f'{source}_d{duty:g}_c{KV}_{arm}_seed{seed}_r{rate:g}.json')


def get_rate(source, duty, arm, seed, rate, cohort, signature, weights):
    path = rate_path(source, duty, arm, seed, rate)
    checkpoint_sha = (run.old.base.digest(run.OUT / run.old.base.read(
        run.OUT / 'checkpoints' / f'seed{seed}_status.json')['selected']['path'])
        if arm == 'rl_mixed' else None)
    if path.exists():
        record = run.old.base.read(path)
        assert record['input_sha256'] == signature
        assert record['checkpoint_sha256'] == checkpoint_sha
        assert record['duty'] == duty
        fresh = False
    else:
        rows = [simulate_one(sample, rate, duty, arm, weights) for sample in cohort]
        rows.sort(key=lambda row: row['name'])
        record = dict(rows=rows, summary=run.old.prior.old.summary(rows, rate),
                      input_sha256=signature, checkpoint_sha256=checkpoint_sha,
                      duty=duty, period_ms=PERIOD_MS, windows=50)
        run.old.base.write(path, record)
        fresh = True
    assert len(record['rows']) == 50
    assert {r['name'] for r in record['rows']} == {s['name'] for s in cohort}
    assert record['summary'] == run.old.prior.old.summary(record['rows'], rate)
    return record['summary'], fresh


def group(task):
    source, duty, arm, seed = task
    target = OUT / 'groups' / f'{source}_d{duty:g}_c{KV}_{arm}_seed{seed}.json'
    if target.exists():
        result = run.old.base.read(target)
        assert result['stage'] == 'complete' and result['duty'] == duty
        return result
    cohort = sweep.cohort_for('original', source)
    signature = sweep.source_hash(cohort)
    weights = run.selected_weights(seed) if arm == 'rl_mixed' else None
    fresh = 0

    def evaluate(rate):
        nonlocal fresh
        summary, new = get_rate(source, duty, arm, seed, rate, cohort, signature, weights)
        fresh += int(new)
        return summary

    passed = 0.
    failed = None
    for rate in COARSE:
        if sweep.passed(evaluate(rate), rate):
            passed = rate
        else:
            failed = rate
            break
    if failed is None:
        raise RuntimeError(f'no failed rate through {COARSE[-1]}: {task}')
    while passed and failed - passed > TARGET_WIDTH + 1e-7:
        rate = round((passed + failed) * 50) / 100
        if not passed < rate < failed:
            break
        if sweep.passed(evaluate(rate), rate):
            passed = rate
        else:
            failed = rate
    result = dict(stage='complete', source=source, duty=duty, arm=arm, seed=seed,
                  kv_blocks=KV, mean_ttft_limit_ms=sweep.LIMIT_MS,
                  period_ms=PERIOD_MS, windows=50, input_sha256=signature,
                  capacity_qps=passed, next_failed_qps=failed,
                  resolution_qps=round(failed - passed, 4) if passed else None,
                  below_minimum_measured_rate=(passed == 0),
                  capacity_summary=evaluate(passed) if passed else None,
                  first_failure_summary=evaluate(failed),
                  newly_simulated_rates=fresh)
    run.old.base.write(target, result)
    return result


def main(sources, duties, arms, seeds, workers):
    tasks = [(source, duty, arm, seed)
             for source in sources for duty in duties for arm in arms
             for seed in ((0,) if arm == 'spare' else seeds)]
    started = time.monotonic()
    done, errors = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(group, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                done.append(result)
                print(f'{len(done)}/{len(tasks)} {task} '
                      f'capacity={result["capacity_qps"]:.2f} '
                      f'new={result["newly_simulated_rates"]}', flush=True)
            except Exception as exc:
                errors.append(dict(task=task, error=repr(exc)))
                print(f'FAILED {task}: {exc!r}', flush=True)
    status = dict(complete=not errors and len(done) == len(tasks),
                  expected=len(tasks), groups=len(done), errors=errors,
                  wall_s=time.monotonic() - started)
    name = (f'status_{"_".join(sources)}_'
            f'{"_".join(str(duty) for duty in duties)}_'
            f'{"_".join(arms)}_{"_".join(str(seed) for seed in seeds)}.json')
    run.old.base.write(OUT / name, status)
    if not status['complete']:
        raise RuntimeError(status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', default='code,conversation')
    parser.add_argument('--duties', default='0.5,0.2')
    parser.add_argument('--arms', default='spare,rl_mixed')
    parser.add_argument('--seeds', default='1017,2027,3037')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    main(tuple(args.sources.split(',')),
         tuple(float(value) for value in args.duties.split(',')),
         tuple(args.arms.split(',')),
         tuple(int(value) for value in args.seeds.split(',')), args.workers)
