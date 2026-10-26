"""Refine selected 2-second mean-TTFT capacity brackets to 0.02 QPS."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from research.www_online_rl_20261017 import analyze_mean, extreme_mean, run
from research.www_kv_extreme_stress_20261016 import run as sim


TASKS = [(arm, source, kv) for kv in (1792, 2048, 2816, 3072, 3840, 4096, 12032)
         for source in ('code', 'conversation') for arm in ('spare', 'rl')]


def one(task):
    import torch
    torch.set_num_threads(1)
    arm, source, kv = task
    suffix = 'arrival' if arm == 'spare' else 'online_rl'
    target = extreme_mean.OUT / 'refined_groups' / f'{source}_c{kv}_{suffix}.json'
    if target.exists():
        result = run.base.read(target)
        assert result['resolution_qps'] <= .020001
        return result
    if arm == 'spare':
        sim.initialize()
    cohort = [sample for sample in run.samples('test') if sample['source'] == source]
    assert len(cohort) == 50
    paths = analyze_mean.records()[arm, source, kv]
    ordered = sorted(paths.items())
    passed_qps, failed_qps = 0., None
    for rate, path in ordered:
        if analyze_mean.passed(run.base.read(path)['summary'], rate,
                               analyze_mean.LIMIT_MS):
            passed_qps = rate
        else:
            failed_qps = rate
            break
    assert failed_qps is not None and failed_qps > passed_qps
    fresh = 0
    while failed_qps - passed_qps > .0200001:
        rate = round((failed_qps + passed_qps) * 50) / 100
        if not passed_qps < rate < failed_qps:
            break
        path = extreme_mean.OUT / 'rates' / f'{source}_c{kv}_{suffix}_r{rate:g}.json'
        if path.exists():
            record = run.base.read(path)
        elif arm == 'spare':
            rows = [sim.simulate_sample(sample, kv, 'arrival', rate) for sample in cohort]
            record = dict(rows=rows, summary=run.prior.old.summary(rows, rate))
            run.base.write(path, record)
            fresh += 1
        else:
            rows = [run.eval_one((sample, kv, rate)) for sample in cohort]
            record = dict(rows=rows, summary=run.prior.old.summary(rows, rate),
                          checkpoint_sha256=run.base.digest(run.OUT / 'checkpoint.json'))
            run.base.write(path, record)
            fresh += 1
        assert len(record['rows']) == 50
        assert {row['name'] for row in record['rows']} == {sample['name'] for sample in cohort}
        assert record['summary'] == run.prior.old.summary(record['rows'], rate)
        if analyze_mean.passed(record['summary'], rate, analyze_mean.LIMIT_MS):
            passed_qps = rate
        else:
            failed_qps = rate
    result = dict(arm=arm, source=source, kv_blocks=kv,
                  mean_ttft_limit_ms=analyze_mean.LIMIT_MS,
                  capacity_qps=passed_qps, next_failed_qps=failed_qps,
                  resolution_qps=round(failed_qps - passed_qps, 4),
                  new_rates=fresh, windows_per_workload=50)
    assert result['resolution_qps'] <= .020001
    run.base.write(target, result)
    return result


def main():
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=12) as pool:
        for future in as_completed([pool.submit(one, task) for task in TASKS]):
            print(future.result(), flush=True)
    run.base.write(extreme_mean.OUT / 'refinement_status.json',
                   dict(complete=True, groups=len(TASKS),
                        wall_s=time.monotonic() - started))


if __name__ == '__main__':
    main()
