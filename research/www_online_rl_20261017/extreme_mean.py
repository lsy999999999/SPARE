"""Evaluate the frozen RL policy at an unseen, high-KV setting."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from research.www_online_rl_20261017 import run
from research.www_mean_ttft_capacity_20261017 import run as mean_run


KV = 12032
OUT = run.ROOT / 'artifacts/www_online_rl_mean_capacity_20261017_r1'


def tasks():
    with (mean_run.OUT / 'rate_audit.csv').open(newline='') as stream:
        audit = list(csv.DictReader(stream))
    result = []
    for source in ('code', 'conversation'):
        rates = sorted({float(row['rate_qps']) for row in audit
                        if row['source'] == source and int(row['kv_blocks']) == KV
                        and row['arm'] == 'arrival'})
        assert rates[0] == .25 and rates[-1] >= 5 and len(rates) >= 15
        result.extend((source, rate) for rate in rates)
    return result


def one(task):
    import torch
    torch.set_num_threads(1)
    source, rate = task
    path = OUT / 'rates' / f'{source}_c{KV}_online_rl_r{rate:g}.json'
    cohort = [sample for sample in run.samples('test') if sample['source'] == source]
    assert len(cohort) == 50
    if path.exists():
        record = run.base.read(path)
    else:
        rows = [run.eval_one((sample, KV, rate)) for sample in cohort]
        record = dict(rows=rows, summary=run.prior.old.summary(rows, rate),
                      checkpoint_sha256=run.base.digest(run.OUT / 'checkpoint.json'))
        run.base.write(path, record)
    assert len(record['rows']) == 50
    assert {row['name'] for row in record['rows']} == {sample['name'] for sample in cohort}
    assert record['summary'] == run.prior.old.summary(record['rows'], rate)
    assert record['checkpoint_sha256'] == run.base.digest(run.OUT / 'checkpoint.json')
    return source, rate, record['summary']['mean_ttft_ms']


def main(workers):
    baseline = mean_run.design()
    original = run.base.read(run.prior.ARCHIVE / 'design.json')
    assert {(s['name'], s['input']) for s in baseline['samples']} == {
        (s['name'], s['input']) for s in original['samples']}
    assert run.base.read(run.OUT / 'checkpoint.json')['training_kv'] == [1024, 2048, 4096, 8192]
    planned = tasks()
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, future in enumerate(as_completed([pool.submit(one, task)
                                                      for task in planned]), 1):
            source, rate, mean = future.result()
            print(f'{index}/{len(planned)} {source} {rate:g} QPS, '
                  f'mean TTFT {mean:.1f} ms', flush=True)
    run.base.write(OUT / 'extreme_status.json', dict(
        complete=True, kv_blocks=KV, rates=len(planned), windows_per_workload=50,
        checkpoint_sha256=run.base.digest(run.OUT / 'checkpoint.json'),
        wall_s=time.monotonic() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    main(args.workers)
