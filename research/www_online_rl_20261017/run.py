"""Train and evaluate an online Q-learning threshold controller on the WWW simulator."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import time
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from research.www_capacity_pareto_20261015 import run as base
from research.www_kv_capacity_fine50_20261016 import run as prior

ROOT = base.ROOT
OUT = ROOT / 'artifacts/www_online_rl_20261017_r1'
ACTIONS = ('spare', 1, 16, 64, 256, 1024)
TRAIN_KV = (1024, 2048, 4096, 8192)
TRAIN_RATES = (.5, 2., 4., 6.)
EVAL_KV = tuple(k for k in prior.CAPACITIES if k >= 1024)


class QPolicy:
    def __init__(self, *, training=False, seed=1017, table=None):
        self.training = training
        self.rng = random.Random(seed)
        self.table = defaultdict(lambda: [0.] * len(ACTIONS))
        self.visits = defaultdict(lambda: [0] * len(ACTIONS))
        if table:
            for state, entry in table.items():
                self.table[state] = list(entry['q'])
                self.visits[state] = list(entry['visits'])
        self.updates = 0
        self.action_counts = [0] * len(ACTIONS)

    def act(self, state):
        visits = self.visits[state]
        if self.training and self.rng.random() < .2:
            action = self.rng.randrange(len(ACTIONS))
        elif self.training:
            action = max(range(len(ACTIONS)), key=lambda i: (self.table[state][i], -visits[i], -i))
        else:
            eligible = [i for i, count in enumerate(visits) if count >= 3]
            action = max(eligible, key=lambda i: (self.table[state][i], -i)) if eligible else 0
        self.action_counts[action] += 1
        return action

    def update(self, state, action, reward, next_state):
        best = max(self.table[next_state])
        old = self.table[state][action]
        self.table[state][action] = old + .05 * (reward + .98 * best - old)
        self.visits[state][action] += 1
        self.updates += 1

    def dump(self):
        return {state: {'q': values, 'visits': self.visits[state]}
                for state, values in sorted(self.table.items())}


def state_of(scheduler):
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    total = scheduler.kv_cache_config.num_blocks - 1
    waiting = list(scheduler.waiting)
    running = list(scheduler.running)
    pending = sum(not r.output_token_ids for r in waiting + running)
    decodes = sum(bool(r.output_token_ids) for r in running)
    free_bin = 0 if free / total < .1 else 1 if free / total < .3 else 2 if free / total < .6 else 3
    pending_bin = 0 if pending <= 1 else 1 if pending <= 3 else 2 if pending <= 8 else 3
    decode_bin = int(decodes >= max(2, len(running) // 2))
    return f'{free_bin}:{pending_bin}:{decode_bin}'


def make_rl_scheduler(policy):
    from research.www_evaluation_completion_20261003.scheduler import EvaluationScheduler

    class RLScheduler(EvaluationScheduler):
        def schedule(self):
            state = state_of(self)
            if self.round_rows and hasattr(self, '_previous'):
                previous = self.round_rows[-1]
                if policy.training:
                    reward = -previous['rl_pending_first_tokens'] * previous['step_ms'] / 1000.
                    policy.update(*self._previous, reward, state)
            action = policy.act(state)
            self.evaluation_options = ({'mode': 'constant', 'n': ACTIONS[action]}
                                       if action else {})
            result = super().schedule()
            self.round_rows[-1]['rl_state'] = state
            self.round_rows[-1]['rl_action'] = action
            self.round_rows[-1]['rl_pending_first_tokens'] = sum(
                not r.output_token_ids for r in self.requests.values())
            self._previous = (state, action)
            return result

    return RLScheduler


def simulate(sample, kv, rate, policy):
    base.initialize()
    sim = base.SIM
    rl_class = make_rl_scheduler(policy)
    original = base.ORIGINAL_MAKE
    capture = {}

    def make(arm='arrival', blocks=768, budget=1024, chunk=256, **kwargs):
        with patch.object(sim, 'FormulaScheduler', rl_class):
            scheduler, clock = original('arrival', blocks, budget, chunk, **kwargs)
        capture['scheduler'] = scheduler
        return scheduler, clock

    class Clock:
        def estimate(self, batch):
            duration, details = base.TIMING.estimate(batch)
            capture['scheduler'].round_rows[-1]['step_ms'] = duration
            return duration, details

    seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
    requests = base.generate(sample, rate)
    with patch.object(sim, 'make_scheduler', make):
        result = sim.simulate(requests, arm='arrival', capacity=kv, budget=1024,
                              chunk=256, history=base.HISTORY[sample['source']],
                              seed=seed, fitted_timing=Clock(), capture_probe=True,
                              max_wall_s=180)
    return result


def samples(split):
    design = base.read(base.OUT / 'design.json')
    return [s for s in design['samples'] if s['split'] == split]


def train():
    import torch
    torch.set_num_threads(1)
    policy = QPolicy(training=True)
    chosen = [s for s in samples('validation') if s['window'] < 16]
    assert len(chosen) == 32
    tasks = [(s, kv, rate) for kv in TRAIN_KV for rate in TRAIN_RATES for s in chosen]
    random.Random(1017).shuffle(tasks)
    started, cpu_started = time.monotonic(), time.process_time()
    failures = []
    for index, (sample, kv, rate) in enumerate(tasks, 1):
        try:
            simulate(sample, kv, rate, policy)
        except RuntimeError as exc:
            failures.append({'sample': sample['name'], 'kv': kv, 'rate': rate, 'error': str(exc)})
        if index % 64 == 0:
            print(f'training {index}/{len(tasks)} episodes, {policy.updates} TD updates', flush=True)
    result = dict(algorithm='tabular Q-learning', training_episodes=len(tasks),
                  train_windows_per_source=16, training_kv=TRAIN_KV, training_rates=TRAIN_RATES,
                  action_thresholds=ACTIONS, state='free KV / pending first tokens / decode majority',
                  reward='negative pending-first-token count times simulated step ms / 1000',
                  epsilon=.2, learning_rate=.05, discount=.98, seed=1017,
                  updates=policy.updates, action_counts=policy.action_counts,
                  failures=failures, wall_s=time.monotonic()-started,
                  cpu_s=time.process_time()-cpu_started, gpu_training=False,
                  q_table=policy.dump())
    base.write(OUT / 'checkpoint.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'q_table'}, indent=2), flush=True)


def eval_one(task):
    sample, kv, rate = task
    import torch
    torch.set_num_threads(1)
    checkpoint = base.read(OUT / 'checkpoint.json')
    policy = QPolicy(table=checkpoint['q_table'])
    started = time.monotonic()
    try:
        result = simulate(sample, kv, rate, policy)
        row = dict(valid=True, **base.probe(result),
                   schedule_mean_ms=result['metrics']['cpu_schedule_including_audit_ms_mean'],
                   rl_actions=policy.action_counts)
    except RuntimeError as exc:
        if str(exc) not in ('empty schedule with pending work; trace/policy cannot progress',
                            'simulation limit; no completed result will be reported'):
            raise
        row = dict(valid=False, failure=str(exc), requests=len(base.generate(sample, rate)))
    return dict(source=sample['source'], name=sample['name'], day=sample['day'],
                capacity=kv, arm='online_rl', budget=1024, rate=rate,
                wall_s=time.monotonic()-started, **row)


def group(task):
    source, kv = task
    cohort = [s for s in samples('test') if s['source'] == source]
    assert len(cohort) == 50
    path = OUT / 'groups' / f'{source}_c{kv}_online_rl.json'
    if path.exists():
        return base.read(path)
    passed, failed = 0., None
    for rate in prior.old.RATES:
        record = rate_record(source, kv, rate, cohort)
        if failed is None:
            if record['summary']['passed']:
                passed = rate
            else:
                failed = rate
        if failed is not None and rate >= .5:
            break
    assert failed is not None and failed > passed
    while failed - passed > .0200001:
        candidate = round((failed + passed) * 50) / 100
        if not passed < candidate < failed:
            break
        record = rate_record(source, kv, candidate, cohort)
        if record['summary']['passed']:
            passed = candidate
        else:
            failed = candidate
    result = dict(source=source, kv_blocks=kv, arm='online_rl', capacity_qps=passed,
                  next_failed_qps=failed, resolution_qps=round(failed-passed, 4),
                  reference_qps=.5, windows_per_workload=50)
    base.write(path, result)
    return result


def rate_record(source, kv, rate, cohort):
    path = OUT / 'rates' / f'{source}_c{kv}_online_rl_r{rate:g}.json'
    if path.exists():
        record = base.read(path)
    else:
        rows = [eval_one((s, kv, rate)) for s in cohort]
        record = dict(rows=rows, summary=prior.old.summary(rows, rate))
        base.write(path, record)
    assert len(record['rows']) == 50
    assert record['summary'] == prior.old.summary(record['rows'], rate)
    return record


def evaluate(workers):
    if not (OUT / 'checkpoint.json').exists():
        raise FileNotFoundError('train checkpoint first')
    design = prior.prepare()
    assert len(design['samples']) == 100
    tasks = [(source, kv) for source in ('code', 'conversation') for kv in EVAL_KV]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, future in enumerate(as_completed([pool.submit(group, task) for task in tasks]), 1):
            future.result()
            if index % 8 == 0 or index == len(tasks):
                print(f'evaluation {index}/{len(tasks)} groups, {time.monotonic()-started:.1f}s', flush=True)
    base.write(OUT / 'evaluation_status.json', dict(complete=True, groups=len(tasks),
                                                    wall_s=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'evaluate'))
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    train() if args.stage == 'train' else evaluate(args.workers)
