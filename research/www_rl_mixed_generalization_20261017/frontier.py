"""Complete KV=2048 SPARE/RL-Mixed capacity scans across mean-TTFT limits."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from research.www_rl_mixed_generalization_20261017 import run, sweep


KV = 2048
LIMITS_MS = (250, 350, 500, 750, 1000, 1500, 2000, 3000,
             4000, 5000, 6000, 8000, 10000, 12000, 16000)
OUT = run.OUT / 'frontier'


def resource_passed(summary, rate):
    return (summary['failed_samples'] == 0
            and summary['p99_tbt_ms'] <= 200
            and summary['backlog_drift'] <= max(2, rate))


def threshold_passed(summary, rate, limit_ms):
    return resource_passed(summary, rate) and summary['mean_ttft_ms'] <= limit_ms


def bisect(last_pass, first_fail, evaluate, predicate):
    while first_fail - last_pass > .0200001:
        rate = round((first_fail + last_pass) * 50) / 100
        if not last_pass < rate < first_fail:
            break
        if predicate(evaluate(rate), rate):
            last_pass = rate
        else:
            first_fail = rate
    return last_pass, first_fail


def scan(task):
    source, arm, seed = task
    target = OUT / 'groups' / f'original_{source}_c{KV}_{arm}_seed{seed}.json'
    if target.exists():
        result = run.old.base.read(target)
        assert result['stage'] == 'complete' and len(result['thresholds']) == len(LIMITS_MS)
        return result
    cohort = sweep.cohort_for('original', source)
    signature = sweep.source_hash(cohort)
    weights = run.selected_weights(seed) if arm == 'rl_mixed' else None
    cached = {}
    fresh = 0

    def evaluate(rate):
        nonlocal fresh
        rate = round(rate, 2)
        if rate not in cached:
            summary, new = sweep.get_rate('original', source, KV, arm, seed,
                                          rate, cohort, signature, weights)
            cached[rate] = summary
            fresh += int(new)
        return cached[rate]

    known = set(sweep.COARSE)
    prefix = f'original_{source}_c{KV}_{arm}_seed{seed}_r'
    for path in (run.OUT / 'rates').glob(prefix + '*.json'):
        known.add(float(path.stem[len(prefix):]))
    if arm == 'spare':
        known.update(sweep.original_records()['spare', source, KV])

    last_resource = 0.
    first_resource = None
    for rate in sorted(known):
        if resource_passed(evaluate(rate), rate):
            last_resource = rate
        else:
            first_resource = rate
            break
    while first_resource is None:
        rate = max(16., last_resource * 2)
        if resource_passed(evaluate(rate), rate):
            last_resource = rate
        else:
            first_resource = rate
        if rate >= 64 and first_resource is None:
            raise RuntimeError(f'no resource failure through {rate:g} QPS: {task}')
    last_resource, first_resource = bisect(
        last_resource, first_resource, evaluate, resource_passed)

    thresholds = []
    minimum_rate = min(known)
    for limit in LIMITS_MS:
        passed = 0.
        failed = first_resource
        for rate in sorted(rate for rate in cached if rate <= first_resource):
            if threshold_passed(evaluate(rate), rate, limit):
                passed = rate
            else:
                failed = rate
                break
        if passed:
            passed, failed = bisect(
                passed, failed, evaluate,
                lambda summary, rate: threshold_passed(summary, rate, limit))
        at_capacity = evaluate(passed) if passed else None
        thresholds.append(dict(mean_ttft_limit_ms=limit, capacity_qps=passed,
                               next_failed_qps=failed,
                               resolution_qps=(round(failed - passed, 4)
                                               if passed else None),
                               below_minimum_measured_rate=(passed == 0),
                               minimum_measured_rate_qps=minimum_rate,
                               attained_mean_ttft_ms=(at_capacity['mean_ttft_ms']
                                                      if at_capacity else None),
                               attained_p99_tbt_ms=(at_capacity['p99_tbt_ms']
                                                    if at_capacity else None),
                               attained_backlog_drift=(at_capacity['backlog_drift']
                                                       if at_capacity else None)))
    result = dict(stage='complete', source=source, arm=arm, seed=seed,
                  kv_blocks=KV, windows=50, input_sha256=signature,
                  resource_last_pass_qps=last_resource,
                  resource_first_fail_qps=first_resource,
                  thresholds=thresholds, rate_records=len(cached), new_rate_records=fresh)
    run.old.base.write(target, result)
    return result


def main(arms, seeds, workers):
    tasks = [(source, arm, seed)
             for source in ('code', 'conversation') for arm in arms
             for seed in ((0,) if arm == 'spare' else seeds)]
    started = time.monotonic()
    done = []
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                done.append(result)
                print(f'{len(done)}/{len(tasks)} {task} '
                      f'rates={result["rate_records"]} '
                      f'new={result["new_rate_records"]} '
                      f'resource={result["resource_last_pass_qps"]:.2f}',
                      flush=True)
            except Exception as exc:
                failures.append(dict(task=task, error=repr(exc)))
                print(f'FAILED {task}: {exc!r}', flush=True)
    status = dict(complete=not failures and len(done) == len(tasks),
                  groups=len(done), expected=len(tasks), failures=failures,
                  wall_s=time.monotonic() - started)
    run.old.base.write(OUT / f'status_{"_".join(arms)}_{"_".join(map(str, seeds))}.json',
                       status)
    if not status['complete']:
        raise RuntimeError(status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', default='spare,rl_mixed')
    parser.add_argument('--seeds', default='1017,2027,3037')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    main(tuple(args.arms.split(',')), tuple(map(int, args.seeds.split(','))), args.workers)
