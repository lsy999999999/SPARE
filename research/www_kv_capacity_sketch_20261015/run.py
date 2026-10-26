"""Fixed paper SPARE versus external adapters, varying only KV capacity."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np

from research.www_capacity_pareto_20261015 import run as prior
from research.www_pareto_papers_20261014.run import LABELS, SCOPES

ROOT = prior.ROOT
OUT = ROOT / 'artifacts/www_kv_capacity_sketch_20261015_r1'
CAPACITIES = (512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120)
SOURCES = ('code', 'conversation')
BUDGET = 1024
REFERENCE_QPS = .5
SLO = dict(p95_ttft_ms=12000., p99_tbt_ms=200.)
read, write, digest = prior.read, prior.write, prior.digest


def initialize():
    import torch
    torch.set_num_threads(1)
    prior.initialize()


def simulate_one(task):
    sample, blocks, arm, rate = task
    seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
    started = time.monotonic()
    requests = prior.generate(sample, rate)
    result = prior.simulate(requests, arm, blocks, BUDGET, sample['source'], seed)
    assert result['metrics']['completed'] == len(requests)
    return dict(source=sample['source'], name=sample['name'], day=sample['day'],
                capacity=blocks, arm=arm, budget=BUDGET, rate=rate,
                valid=True, **prior.probe(result), wall_s=time.monotonic()-started)


def summary(rows, rate):
    assert len(rows) == 2 and len({row['name'] for row in rows}) == 2
    assert all(row['valid'] for row in rows)
    result = {key: float(np.mean([row[key] for row in rows])) for key in
              ('mean_ttft_ms', 'p95_ttft_ms', 'p99_tbt_ms', 'backlog_drift')}
    result['drift_limit'] = max(2., rate)
    result['passed'] = (result['p95_ttft_ms'] <= SLO['p95_ttft_ms']
                        and result['p99_tbt_ms'] <= SLO['p99_tbt_ms']
                        and result['backlog_drift'] <= result['drift_limit'])
    return result


def rate_path(source, blocks, arm, rate):
    return OUT / 'rates' / f'{source}_c{blocks}_{arm}_r{rate:g}.json'


def prepare():
    old = read(ROOT / 'artifacts/www_kv_capacity_paper_20261015_r1/design.json')
    for rel, sha in old['source_sha256'].items():
        assert digest(ROOT / rel) == sha, rel
    assert digest(prior.PREVIOUS / 'timing_profile.json') == old['timing_sha256']
    samples = []
    for source in SOURCES:
        days = sorted({s['day'] for s in old['samples'] if s['source'] == source})
        for day in days:
            samples.append(min((s for s in old['samples'] if s['source'] == source and s['day'] == day), key=lambda s: s['window']))
    for s in samples:
        assert digest(ROOT / s['input']) == old['input_sha256'][s['input']]
    design = dict(old, samples=samples, capacities=CAPACITIES, windows_per_source=2,
                  draft=True, sample_selection='First test window from each date, fixed before running',
                  endpoint_reuse='Subset of previous raw trajectories, never previous 50-window means')
    write(OUT / 'design.json', design)
    return samples


def reuse_rates(samples):
    old_out = ROOT / 'artifacts/www_kv_capacity_paper_20261015_r1'
    for source in SOURCES:
        names = {s['name'] for s in samples if s['source'] == source}
        for blocks in CAPACITIES:
            for arm in prior.MAIN:
                for rate in prior.RATES:
                    target = rate_path(source, blocks, arm, rate)
                    if target.exists():
                        continue
                    path = old_out / 'rates' / f'{source}_c{blocks}_{arm}_r{rate:g}.json'
                    if path.exists():
                        rows = [r for r in read(path)['rows'] if r['name'] in names]
                        assert {r['name'] for r in rows} == names
                        write(target, dict(reused=True, rows=rows, summary=summary(rows, rate)))


def run(jobs):
    samples = prepare()
    started = time.monotonic()
    write(OUT / 'status.json', dict(stage='initializing', pid=os.getpid()))
    with ProcessPoolExecutor(max_workers=jobs, initializer=initialize) as pool:
        reuse_rates(samples)
        groups = [(source, blocks, arm) for source in SOURCES
                  for blocks in CAPACITIES for arm in prior.MAIN]
        pending, batches, completed_groups = {}, defaultdict(list), set()
        completed_new = 0
        completed_cached = 0
        errors = []

        def advance(group, index=0, already_failed=False):
            nonlocal completed_cached
            source, blocks, arm = group
            while index < len(prior.RATES):
                rate = prior.RATES[index]
                path = rate_path(*group, rate)
                if path.exists():
                    data = read(path)
                    assert data['summary'] == summary(data['rows'], rate)
                    completed_cached += len(data['rows'])
                    already_failed |= not data['summary']['passed']
                    if already_failed and rate >= REFERENCE_QPS:
                        completed_groups.add(group)
                        return
                    index += 1
                    continue
                for sample in samples:
                    if sample['source'] == source:
                        task = (sample, blocks, arm, rate)
                        pending[pool.submit(simulate_one, task)] = (group, index, already_failed)
                return
            completed_groups.add(group)

        for group in groups:
            advance(group)

        def status(stage='running'):
            result = dict(stage=stage, pid=os.getpid(), jobs=jobs,
                          groups_complete=len(completed_groups), groups_total=len(groups),
                          new_simulations=completed_new, reused_simulations=completed_cached,
                          pending_simulations=len(pending),
                          elapsed_s=round(time.monotonic()-started, 1), errors=errors)
            write(OUT / 'status.json', result)
            return result

        last_update = 0.
        while pending:
            done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                group, index, already_failed = pending.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    errors.append(dict(group=group, rate=prior.RATES[index], error=repr(exc)))
                    status('failed')
                    raise
                completed_new += 1
                key = group, index
                batches[key].append(row)
                if len(batches[key]) == 2:
                    rows = sorted(batches.pop(key), key=lambda row: row['name'])
                    rate = prior.RATES[index]
                    stats = summary(rows, rate)
                    write(rate_path(*group, rate), dict(reused=False, rows=rows, summary=stats))
                    failed = already_failed or not stats['passed']
                    if failed and rate >= REFERENCE_QPS:
                        completed_groups.add(group)
                    else:
                        advance(group, index+1, failed)
            if time.monotonic()-last_update >= 20:
                print(status(), flush=True)
                last_update = time.monotonic()
        assert len(completed_groups) == len(groups)
        print(status('simulations_complete'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=24)
    run(parser.parse_args().jobs)
