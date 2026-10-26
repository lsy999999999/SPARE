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
OUT = ROOT / 'artifacts/www_kv_capacity_paper_20261015_r1'
CAPACITIES = (768, 1024, 1536, 2048, 3072, 4096)
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
    assert len(rows) == 50 and len({row['name'] for row in rows}) == 50
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
    old = read(prior.OUT / 'design.json')
    samples = [s for s in old['samples'] if s['split'] == 'test']
    assert all(sum(s['source'] == source for s in samples) == 50 for source in SOURCES)
    for rel, sha in old['source_inputs'].items():
        assert digest(ROOT / rel) == sha, rel
    assert digest(prior.PREVIOUS / 'timing_profile.json') == old['timing_sha256']
    files = {str(path.relative_to(ROOT)): digest(path)
             for folder in ('research/www_evaluation_completion_20261003',
                            'research/www_evaluation_gpu_completion_20261003',
                            'research/v1_budget_20260922',
                            'research/v1_paper_comparison_20260922',
                            'research/www_legacy_baselines_20261006',
                            'research/www_baseline_fidelity_20261013',
                            'research/v1_broad12h_20260925',
                            'research/v1_saving_threshold_20260924')
             for path in (ROOT / folder).glob('*.py')}
    design = dict(samples=samples, main_arms=prior.MAIN, labels=LABELS,
                  adapter_scopes={arm: SCOPES[arm] for arm in prior.MAIN},
                  capacities=CAPACITIES, default_budget=BUDGET, chunk=256,
                  concurrency=32, block_tokens=16, reference_qps=REFERENCE_QPS,
                  slos=SLO, rates=prior.RATES, windows_per_source=50,
                  horizon_ms=60000, warm_ms=20000, backlog_interval_ms=[40000, 60000],
                  backlog_limit='max(2, offered QPS) requests over the last 20 seconds',
                  policy='Paper SPARE, n0=8, coefficient=128, first passing candidate',
                  alpha_selection=False, gpu_execution=False,
                  aggregation='Equal mean of Code and Conversation; not mixed-stream capacity',
                  scan='Ascending QPS grid through first ensemble failure; no rates skipped below it',
                  endpoint_reuse=str(prior.OUT), timing_sha256=old['timing_sha256'],
                  source_sha256=files, input_sha256=old['source_inputs'])
    path = OUT / 'design.json'
    if path.exists():
        assert read(path) == read_normalized(design), 'Frozen design changed'
    else:
        write(path, design)
        for rel in files:
            target = OUT / 'source' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / rel).read_bytes())
    return samples


def read_normalized(value):
    import json
    return json.loads(json.dumps(value))


def audit_reuse(pool, samples):
    path = OUT / 'reuse_audit.json'
    if path.exists() and read(path).get('passed'):
        return
    tasks = [(next(s for s in samples if s['source'] == source), blocks, arm, 1.5)
             for source in SOURCES for blocks in (768, 4096) for arm in prior.MAIN]
    futures = {pool.submit(simulate_one, task): task for task in tasks}
    records = []
    for future in futures:
        sample, blocks, arm, rate = futures[future]
        actual = future.result()
        old = next(row for row in read(prior.OUT / 'test' /
                   f"{sample['name']}_c{blocks}_{arm}_t{BUDGET}.json") if row['rate'] == rate)
        differences = [key for key, value in old.items()
                       if key not in ('wall_s',) and actual.get(key) != value]
        records.append(dict(name=sample['name'], blocks=blocks, arm=arm,
                            rate=rate, differences=differences,
                            wall_s=actual['wall_s']))
    passed = all(not row['differences'] for row in records)
    write(path, dict(passed=passed, cases=records))
    assert passed, 'Cached endpoint trajectories did not reproduce'
    print(f'Reuse audit passed: {len(records)} paired trajectories', flush=True)


def reuse_rates(samples):
    for source in SOURCES:
        selected = [s for s in samples if s['source'] == source]
        for blocks in (768, 4096):
            for arm in prior.MAIN:
                by_rate = defaultdict(list)
                for sample in selected:
                    for row in read(prior.OUT / 'test' /
                                    f"{sample['name']}_c{blocks}_{arm}_t{BUDGET}.json"):
                        by_rate[row['rate']].append(row)
                failed = False
                for rate in prior.RATES:
                    rows = sorted(by_rate[rate], key=lambda row: row['name'])
                    stats = summary(rows, rate)
                    write(rate_path(source, blocks, arm, rate),
                          dict(reused=True, rows=rows, summary=stats))
                    failed = failed or not stats['passed']
                    if failed and rate >= REFERENCE_QPS:
                        break


def run(jobs):
    samples = prepare()
    started = time.monotonic()
    write(OUT / 'status.json', dict(stage='reuse_audit', pid=os.getpid()))
    with ProcessPoolExecutor(max_workers=jobs, initializer=initialize) as pool:
        audit_reuse(pool, samples)
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
                if len(batches[key]) == 50:
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
