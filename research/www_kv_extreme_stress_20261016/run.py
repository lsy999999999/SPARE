"""Resumable paired KV capacity sweeps, including a separate short-request cohort."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import math
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

from research.www_capacity_pareto_20261015 import run as base
from research.www_pareto_papers_20261014.run import LABELS, SCOPES

ROOT = base.ROOT
OUT = ROOT / 'artifacts/www_kv_extreme_stress_20261016_r1'
OLD_PAPER = ROOT / 'artifacts/www_kv_capacity_paper_20261015_r1'
OLD_SKETCH = ROOT / 'artifacts/www_kv_capacity_sketch_20261015_r1'
METHODS = ('arrival', 'native', 'sarathi', 'sola_rule',
           'uniboost_static', 'past_future', 's3_proxy', 'apt_serve_kv_rule')
CAPACITIES = {
    'full': (512, 528, 544, 560, 576, 608, 640, 672, 704, 736, 768,
             832, 896, 1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192),
    'short': (128, 144, 160, 192, 224, 256, 320, 384, 448, 512, 640,
              768, 1024, 1536, 2048, 3072, 4096),
}
RATES = {
    'full': tuple(base.RATES),
    'short': tuple(base.RATES) + (12., 16., 20., 24.),
}
SHORT_MIN_BLOCKS = 128
SLO = dict(p95_ttft_ms=12000., p99_tbt_ms=200.)
BUDGET = 1024
REFERENCE_QPS = .5


def choose_samples():
    design = base.read(base.OUT / 'design.json')
    samples = [s for s in design['samples']
               if s['split'] == 'test' and s['window'] in (0, 5, 10, 15, 20)]
    assert len(samples) == 20
    for source in ('code', 'conversation'):
        group = [s for s in samples if s['source'] == source]
        assert len(group) == 10 and len({s['day'] for s in group}) == 2
    return sorted(samples, key=lambda s: s['name']), design


def prepare():
    samples, old = choose_samples()
    assert base.digest(base.PREVIOUS / 'timing_profile.json') == old['timing_sha256']
    assert set(METHODS) <= set(base.MAIN)
    result = {'full': [], 'short': []}
    selection = []
    for sample in samples:
        source = ROOT / sample['input']
        assert base.digest(source) == old['source_inputs'][sample['input']]
        lengths = base.read(source)
        max_blocks = max(math.ceil((r['prompt'] + r['output'] - 1) / 16) for r in lengths)
        assert max_blocks <= CAPACITIES['full'][0] - 1
        result['full'].append(sample)
        filtered = [r for r in lengths
                    if math.ceil((r['prompt'] + r['output'] - 1) / 16) <= SHORT_MIN_BLOCKS - 1]
        assert len(filtered) >= 16
        short_path = OUT / 'inputs' / f"{sample['name']}_short.json"
        if short_path.exists():
            assert base.read(short_path) == filtered
        else:
            base.write(short_path, filtered)
        result['short'].append(dict(sample, input=str(short_path.relative_to(ROOT))))
        selection.append(dict(name=sample['name'], source=sample['source'], day=sample['day'],
                              full_requests=len(lengths), short_requests=len(filtered),
                              max_full_blocks=max_blocks,
                              max_short_blocks=max(math.ceil((r['prompt']+r['output']-1)/16)
                                                   for r in filtered),
                              full_sha256=base.digest(source), short_sha256=base.digest(short_path)))
    for cohort in ('full', 'short'):
        target = OUT / cohort / 'design.json'
        design = dict(cohort=cohort, samples=result[cohort],
                      capacities=CAPACITIES[cohort], rates=RATES[cohort],
                      methods=METHODS, adapter_scopes={m: SCOPES[m] for m in METHODS},
                      labels={m: LABELS[m] for m in METHODS},
                      budget=BUDGET, chunk=256, concurrency=32, block_tokens=16,
                      short_min_blocks=SHORT_MIN_BLOCKS if cohort == 'short' else None,
                      reference_qps=REFERENCE_QPS, slo=SLO,
                      horizon_ms=60000., warm_ms=20000.,
                      backlog_interval_ms=[40000., 60000.],
                      backlog_limit='max(2, offered QPS) outstanding requests over 20 s',
                      sample_selection='Five fixed test windows per date and source: 0,5,10,15,20',
                      selection=selection,
                      timing_sha256=base.digest(base.PREVIOUS / 'timing_profile.json'),
                      history_sha256=base.digest(base.PREVIOUS / 'history.json'),
                      cost_sha256=base.digest(ROOT / 'artifacts/www_baseline_fidelity_20261013_r1/calibration/cost_model.json'),
                      input_sha256={s['input']: base.digest(ROOT / s['input']) for s in result[cohort]},
                      gpu_execution=False,
                      source_sha256={p: base.digest(ROOT / p) for p in (
                          'research/www_kv_extreme_stress_20261016/run.py',
                          'research/www_capacity_pareto_20261015/run.py',
                          'research/www_evaluation_completion_20261003/simulate.py',
                          'research/www_pareto_papers_20261014/run.py')})
        if target.exists():
            existing = base.read(target)
            normalized = __import__('json').loads(__import__('json').dumps(design))
            if existing != normalized:
                old_sources = existing.pop('source_sha256')
                new_sources = normalized.pop('source_sha256')
                assert existing == normalized
                own = 'research/www_kv_extreme_stress_20261016/run.py'
                assert {k: v for k, v in old_sources.items() if k != own} == {
                    k: v for k, v in new_sources.items() if k != own}
                base.write(target, design)
        else:
            base.write(target, design)
    base.write(OUT / 'selection.json', selection)
    return result


def validate_design(cohort):
    d = base.read(OUT / cohort / 'design.json')
    assert tuple(d['capacities']) == CAPACITIES[cohort]
    assert tuple(d['rates']) == RATES[cohort]
    assert tuple(d['methods']) == METHODS
    for path, sha in d['input_sha256'].items():
        assert base.digest(ROOT / path) == sha, path
    for path, sha in d['source_sha256'].items():
        assert base.digest(ROOT / path) == sha, path
    assert base.digest(base.PREVIOUS / 'timing_profile.json') == d['timing_sha256']
    assert base.digest(base.PREVIOUS / 'history.json') == d['history_sha256']
    assert base.digest(ROOT / 'artifacts/www_baseline_fidelity_20261013_r1/calibration/cost_model.json') == d['cost_sha256']
    return d


def initialize():
    import torch
    torch.set_num_threads(1)
    base.initialize()
    base.MAX_WALL_S = 180


def summary(rows, rate):
    assert len(rows) == 10 and len({r['name'] for r in rows}) == 10
    failures = [r for r in rows if not r['valid']]
    valid = [r for r in rows if r['valid']]
    stats = {key: float(np.mean([r[key] for r in rows])) for key in (
        'mean_ttft_ms', 'p95_ttft_ms', 'p99_tbt_ms', 'backlog_drift',
        'preemptions', 'recomputed_tokens')} if not failures else {
            key: float(np.mean([r[key] for r in valid])) if valid else None
            for key in ('mean_ttft_ms', 'p95_ttft_ms', 'p99_tbt_ms', 'backlog_drift',
                        'preemptions', 'recomputed_tokens')}
    stats['drift_limit'] = max(2., rate)
    if failures:
        stats['failed_samples'] = len(failures)
        stats['failure_reasons'] = sorted({r['failure'] for r in failures})
        stats['passed'] = False
    else:
        stats['passed'] = (stats['p95_ttft_ms'] <= SLO['p95_ttft_ms'] and
                           stats['p99_tbt_ms'] <= SLO['p99_tbt_ms'] and
                           stats['backlog_drift'] <= stats['drift_limit'])
    return stats


def cached_rows(source, blocks, arm, rate, wanted):
    if blocks in (768, 1024, 1536, 2048, 3072, 4096):
        path = OLD_PAPER / 'rates' / f'{source}_c{blocks}_{arm}_r{rate:g}.json'
        if path.exists():
            rows = [r for r in base.read(path)['rows'] if r['name'] in wanted]
            if {r['name'] for r in rows} == wanted:
                return rows
    path = OLD_SKETCH / 'rates' / f'{source}_c{blocks}_{arm}_r{rate:g}.json'
    if path.exists():
        return [r for r in base.read(path)['rows'] if r['name'] in wanted]
    return []


def simulate_sample(sample, blocks, arm, rate):
    seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
    requests = base.generate(sample, rate)
    try:
        output = base.simulate(requests, arm, blocks, BUDGET, sample['source'], seed)
    except RuntimeError as exc:
        if str(exc) not in ('empty schedule with pending work; trace/policy cannot progress',
                            'simulation limit; no completed result will be reported'):
            raise
        return dict(source=sample['source'], name=sample['name'], day=sample['day'],
                    capacity=blocks, arm=arm, budget=BUDGET, rate=rate,
                    valid=False, failure=str(exc), requests=len(requests))
    assert output['metrics']['completed'] == len(requests)
    return dict(source=sample['source'], name=sample['name'], day=sample['day'],
                capacity=blocks, arm=arm, budget=BUDGET, rate=rate, valid=True,
                **base.probe(output))


def run_group(task):
    cohort, source, blocks, arm, samples = task
    wanted = {s['name'] for s in samples}
    assert len(wanted) == 10
    passed_capacity, failed_rate, reused, new, rates_done = 0., None, 0, 0, 0
    for rate in RATES[cohort]:
        path = OUT / cohort / 'rates' / f'{source}_c{blocks}_{arm}_r{rate:g}.json'
        if path.exists():
            record = base.read(path)
            assert {r['name'] for r in record['rows']} == wanted
            assert record['summary'] == summary(record['rows'], rate)
            rows = record['rows']
        else:
            rows = cached_rows(source, blocks, arm, rate, wanted) if cohort == 'full' else []
            found = {r['name'] for r in rows}
            assert found <= wanted
            reused += len(rows)
            for sample in samples:
                if sample['name'] not in found:
                    rows.append(simulate_sample(sample, blocks, arm, rate))
                    new += 1
            rows.sort(key=lambda r: r['name'])
            record = dict(rows=rows, summary=summary(rows, rate),
                          reused_rows=len(found), newly_simulated=len(rows)-len(found))
            base.write(path, record)
        rates_done += 1
        if failed_rate is None:
            if record['summary']['passed']:
                passed_capacity = rate
            else:
                failed_rate = rate
        if failed_rate is not None and rate >= REFERENCE_QPS:
            break
    result = dict(cohort=cohort, source=source, capacity=blocks, arm=arm,
                  capacity_qps=passed_capacity, first_failed_qps=failed_rate,
                  capacity_is_lower_bound=failed_rate is None,
                  reference_qps=REFERENCE_QPS, reference_feasible=passed_capacity >= REFERENCE_QPS,
                  rates_done=rates_done, reused_rows=reused, new_rows=new)
    base.write(OUT / cohort / 'groups' / f'{source}_c{blocks}_{arm}.json', result)
    return result


def run(cohort, jobs, limit_groups=None):
    design = validate_design(cohort)
    tasks = [(cohort, source, capacity, arm,
              [s for s in design['samples'] if s['source'] == source])
             for source in ('code', 'conversation')
             for capacity in CAPACITIES[cohort] for arm in METHODS]
    if limit_groups is not None:
        tasks = tasks[:limit_groups]
    target = OUT / cohort
    started = time.monotonic()
    completed = 0
    status = dict(stage='running', host=os.uname().nodename, cohort=cohort,
                  pid=os.getpid(), groups_total=len(tasks), jobs=jobs,
                  groups_complete=0, new_rows=0, reused_rows=0,
                  rates_complete=0, errors=[], updated_at=time.time())
    base.write(target / 'status.json', status)
    with ProcessPoolExecutor(max_workers=jobs, initializer=initialize) as pool:
        future_map = {pool.submit(run_group, task): task for task in tasks}
        for future in as_completed(future_map):
            try:
                row = future.result()
            except Exception as exc:
                task = future_map[future]
                status['errors'].append(dict(source=task[1], capacity=task[2],
                                             arm=task[3], error=repr(exc)))
                status['stage'] = 'failed'
                status['updated_at'] = time.time()
                base.write(target / 'status.json', status)
                raise
            completed += 1
            status['groups_complete'] = completed
            status['new_rows'] += row['new_rows']
            status['reused_rows'] += row['reused_rows']
            status['rates_complete'] += row['rates_done']
            status['elapsed_s'] = round(time.monotonic()-started, 1)
            status['updated_at'] = time.time()
            if completed % 4 == 0 or completed == len(tasks):
                base.write(target / 'status.json', status)
                print(status, flush=True)
    status['stage'] = 'complete'
    base.write(target / 'status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--cohort', choices=('full', 'short'))
    parser.add_argument('--jobs', type=int, default=12)
    parser.add_argument('--limit-groups', type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        if not args.cohort:
            parser.error('--cohort is required unless --prepare is used')
        run(args.cohort, args.jobs, args.limit_groups)
