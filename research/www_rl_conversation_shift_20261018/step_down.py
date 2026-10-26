"""Paired load-step experiment: 4.5 QPS falls to 1/2 or 1/5 at 30 s."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from statistics import mean
import time
from unittest.mock import patch

import numpy as np

from research.www_rl_conversation_shift_20261018 import run as experiment
from research.www_rl_mixed_generalization_20261017 import run as mixed, sweep


OUT = experiment.OUT / 'step_down_20261018'
PRE_QPS = 4.5
SWITCH_MS = 30_000.
HORIZON_MS = 60_000.
RATIOS = (1., .5, .2)
SEEDS = experiment.SEEDS
KNOWN_FAILURES = ('empty schedule with pending work; trace/policy cannot progress',
                  'simulation limit; no completed result will be reported')


def configure():
    mixed.OUT = experiment.OUT
    for seed in SEEDS:
        status = mixed.old.base.read(
            experiment.OUT / 'checkpoints' / f'seed{seed}_status.json')
        assert status['train_source'] == 'conversation'
        assert status['train_kv_blocks'] == experiment.KV


def requests_for(sample, ratio):
    base = mixed.old.base.generate(sample, PRE_QPS)
    requests = []
    for row in base:
        arrival = row['arrival_ms']
        shifted = (arrival if arrival < SWITCH_MS else
                   SWITCH_MS + (arrival - SWITCH_MS) / ratio)
        if shifted < HORIZON_MS:
            requests.append(dict(row, arrival_ms=shifted))
    assert all(a['arrival_ms'] <= b['arrival_ms'] for a, b in
               zip(requests, requests[1:]))
    assert all(a['arrival_ms'] == b['arrival_ms'] for a, b in
               zip(requests, base) if a['arrival_ms'] < SWITCH_MS)
    return requests


def stream_hash(requests):
    content = json.dumps(requests, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(content.encode()).hexdigest()


def summarize(result):
    records = result['probe_records']

    def phase(start, end):
        chosen = [r for r in records if start <= r['arrival_ms'] < end]
        assert chosen
        values = [r['ttft_ms'] for r in chosen]
        return dict(requests=len(chosen), mean_ttft_ms=mean(values),
                    p95_ttft_ms=float(np.percentile(values, 95)))

    def outstanding(at):
        return sum(r['arrival_ms'] <= at < r['finish_ms'] for r in records)

    pre = phase(20_000., SWITCH_MS)
    post = phase(SWITCH_MS, HORIZON_MS)
    late = phase(45_000., HORIZON_MS)
    return dict(pre=pre, post=post, late=late,
                outstanding_at_30s=outstanding(SWITCH_MS),
                outstanding_at_45s=outstanding(45_000.),
                outstanding_at_60s=outstanding(HORIZON_MS),
                backlog_change_30_60=outstanding(HORIZON_MS) -
                outstanding(SWITCH_MS),
                p99_tbt_ms=mixed.old.base.probe(result)['p99_tbt_ms'])


def simulate_sample(sample, ratio, arm, weights):
    requests = requests_for(sample, ratio)
    signature = stream_hash(requests)
    try:
        if arm == 'spare':
            if not hasattr(mixed.old.base, 'SIM'):
                mixed.old.base.initialize()
            seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4],
                                  'big')
            result = mixed.old.base.simulate(
                requests, 'arrival', experiment.KV, 1024, sample['source'], seed)
        else:
            actor = experiment.DQNPolicy(weights=weights)
            with patch.object(mixed.old.base, 'generate', return_value=requests):
                result = mixed.simulate(sample, experiment.KV, PRE_QPS, actor)
        metrics = summarize(result)
        return dict(name=sample['name'], valid=True, input_sha256=signature,
                    arrivals=len(requests), **metrics)
    except RuntimeError as exc:
        if str(exc) not in KNOWN_FAILURES:
            raise
        return dict(name=sample['name'], valid=False, input_sha256=signature,
                    arrivals=len(requests), failure=str(exc))


def group(task):
    ratio, arm, seed = task
    path = OUT / 'groups' / f'conversation_ratio{ratio:g}_{arm}_seed{seed}.json'
    if path.exists():
        data = mixed.old.base.read(path)
        assert data['ratio'] == ratio and data['arm'] == arm and data['seed'] == seed
        return data
    cohort = sweep.cohort_for('original', 'conversation')
    assert len(cohort) == 50
    weights = mixed.selected_weights(seed) if arm == 'rl_mixed' else None
    rows = [simulate_sample(sample, ratio, arm, weights) for sample in cohort]
    rows.sort(key=lambda row: row['name'])
    valid = [row for row in rows if row['valid']]
    fields = ('pre', 'post', 'late')
    summary = dict(valid_windows=len(valid), failed_windows=len(rows) - len(valid),
                   mean_arrivals=mean(row['arrivals'] for row in rows))
    if valid:
        for field in fields:
            summary[f'{field}_mean_ttft_ms'] = mean(
                row[field]['mean_ttft_ms'] for row in valid)
            summary[f'{field}_p95_ttft_ms'] = mean(
                row[field]['p95_ttft_ms'] for row in valid)
            summary[f'{field}_mean_requests'] = mean(
                row[field]['requests'] for row in valid)
        for field in ('outstanding_at_30s', 'outstanding_at_45s',
                      'outstanding_at_60s', 'backlog_change_30_60', 'p99_tbt_ms'):
            summary[f'mean_{field}'] = mean(row[field] for row in valid)
    data = dict(stage='complete', source='conversation', kv_blocks=experiment.KV,
                pre_qps=PRE_QPS, post_qps=PRE_QPS * ratio, switch_ms=SWITCH_MS,
                horizon_ms=HORIZON_MS, ratio=ratio, arm=arm, seed=seed,
                input_sha256=sweep.source_hash(cohort),
                stream_sha256=hashlib.sha256(''.join(
                    row['input_sha256'] for row in rows).encode()).hexdigest(),
                windows=50, summary=summary, rows=rows)
    mixed.old.base.write(path, data)
    return data


def main(workers):
    configure()
    tasks = [(ratio, arm, seed) for ratio in RATIOS
             for arm in ('spare', 'rl_mixed')
             for seed in ((0,) if arm == 'spare' else SEEDS)]
    started = time.monotonic()
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(group, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                result = future.result()
                print(f'{index}/{len(tasks)} ratio={task[0]:g} arm={task[1]} '
                      f'seed={task[2]} post_ttft_ms='
                      f'{result["summary"].get("post_mean_ttft_ms", float("nan")):.1f}',
                      flush=True)
            except Exception as exc:
                errors.append(dict(task=task, error=repr(exc)))
                print(f'FAILED {task}: {exc!r}', flush=True)
    status = dict(complete=not errors, expected=len(tasks),
                  completed=len(tasks) - len(errors), errors=errors,
                  wall_s=time.monotonic() - started)
    mixed.old.base.write(OUT / 'status.json', status)
    if errors:
        raise RuntimeError(status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    main(args.workers)
