"""Train one mixed-condition policy and replay it on paired simulator windows."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import torch

from research.www_online_rl_20261017 import run as old
from research.www_rl_mixed_generalization_20261017 import inputs, policy as learned


OUT = inputs.OUT
TRAIN_KV = (1024, 2048, 4096, 8192)
TRAIN_RATES = (.5, 2., 4., 6.)
CHECKPOINT_EPISODES = (512, 1024)


def make_scheduler(policy):
    from research.www_evaluation_completion_20261003.scheduler import EvaluationScheduler

    class LearnedScheduler(EvaluationScheduler):
        def schedule(self):
            state = learned.state_of(self)
            if self.round_rows and hasattr(self, '_rl_previous') and policy.training:
                previous = self.round_rows[-1]
                policy.observe(*self._rl_previous, learned.reward_of(previous), state)
            action = policy.act(state)
            self.evaluation_options = ({'mode': 'constant', 'n': learned.ACTIONS[action]}
                                       if action else {})
            result = super().schedule()
            self.round_rows[-1]['rl_pending_first_tokens'] = sum(
                not r.output_token_ids for r in self.requests.values())
            self.round_rows[-1]['rl_decode_count'] = sum(
                bool(r.output_token_ids) for r in self.running)
            self.round_rows[-1]['rl_action'] = action
            self._rl_previous = (state, action)
            return result

    return LearnedScheduler


def simulate(sample, kv, rate, policy):
    if not hasattr(old.base, 'SIM'):
        old.base.initialize()
    sim = old.base.SIM
    scheduler_class = make_scheduler(policy)
    capture = {}

    def make(arm='arrival', blocks=768, budget=1024, chunk=256, **kwargs):
        with patch.object(sim, 'FormulaScheduler', scheduler_class):
            scheduler, clock = old.base.ORIGINAL_MAKE(
                'arrival', blocks, budget, chunk, **kwargs)
        capture['scheduler'] = scheduler
        return scheduler, clock

    class Clock:
        def estimate(self, batch):
            duration, details = old.base.TIMING.estimate(batch)
            capture['scheduler'].round_rows[-1]['step_ms'] = duration
            return duration, details

    seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
    requests = old.base.generate(sample, rate)
    with patch.object(sim, 'make_scheduler', make):
        result = sim.simulate(requests, arm='arrival', capacity=kv, budget=1024,
                              chunk=256, history=old.base.HISTORY[sample['source']],
                              seed=seed, fitted_timing=Clock(), capture_probe=True,
                              max_wall_s=180)
    scheduler = capture['scheduler']
    if policy.training and hasattr(scheduler, '_rl_previous'):
        last = scheduler.round_rows[-1]
        policy.observe(*scheduler._rl_previous, learned.reward_of(last),
                       learned.state_of(scheduler), done=True)
    return result


def eval_one(sample, kv, rate, weights):
    torch.set_num_threads(1)
    actor = learned.DQNPolicy(weights=weights)
    started = time.monotonic()
    try:
        result = simulate(sample, kv, rate, actor)
        row = dict(valid=True, **old.base.probe(result),
                   schedule_mean_ms=result['metrics']['cpu_schedule_including_audit_ms_mean'],
                   rl_actions=actor.action_counts)
    except RuntimeError as exc:
        if str(exc) not in ('empty schedule with pending work; trace/policy cannot progress',
                            'simulation limit; no completed result will be reported'):
            raise
        row = dict(valid=False, failure=str(exc),
                   requests=len(old.base.generate(sample, rate)))
    return dict(source=sample['source'], name=sample['name'], day=sample['day'],
                capacity=kv, arm='rl_mixed', budget=1024, rate=rate,
                wall_s=time.monotonic() - started, **row)


def dev_score(weights):
    heldout = [s for s in old.samples('validation') if s['window'] >= 16]
    assert len(heldout) == 8
    conditions = [(kv, rate) for kv in (2048, 4096) for rate in (2., 4.)]
    results = []
    for kv, rate in conditions:
        rows = [eval_one(sample, kv, rate, weights) for sample in heldout]
        valid = [row for row in rows if row['valid']]
        failed = len(rows) - len(valid)
        mean_ttft = (sum(row['mean_ttft_ms'] for row in valid) / len(valid)
                     if valid else 1e6)
        p99_tbt = (sum(row['p99_tbt_ms'] for row in valid) / len(valid)
                   if valid else 1e6)
        backlog = (sum(row['backlog_drift'] for row in valid) / len(valid)
                   if valid else 1e6)
        score = math.log1p(mean_ttft)
        score += 2 * failed
        score += max(0, p99_tbt - 200) / 200
        score += max(0, backlog - max(2, rate)) / max(2, rate)
        results.append(dict(kv_blocks=kv, rate_qps=rate, score=score,
                            mean_ttft_ms=mean_ttft, p99_tbt_ms=p99_tbt,
                            backlog_drift=backlog, failed_samples=failed))
    return sum(r['score'] for r in results) / len(results), results


def train(seed, episodes):
    assert episodes in CHECKPOINT_EPISODES
    torch.set_num_threads(1)
    resumed_path = OUT / 'checkpoints' / f'seed{seed}_ep512.pt'
    resume = episodes == 1024 and resumed_path.exists()
    starting_weights = (torch.load(resumed_path, map_location='cpu', weights_only=True)
                        if resume else None)
    actor = learned.DQNPolicy(training=True, seed=seed, weights=starting_weights)
    if resume:
        actor.steps = 1000000
    chosen = [s for s in old.samples('validation') if s['window'] < 16]
    assert len(chosen) == 32
    all_tasks = []
    base_tasks = [(s, kv, rate) for s in chosen
                  for kv in TRAIN_KV for rate in TRAIN_RATES]
    for epoch in range(math.ceil(episodes / len(base_tasks))):
        tasks = base_tasks.copy()
        random.Random(seed + epoch).shuffle(tasks)
        all_tasks.extend(tasks)
    start = time.monotonic()
    checkpoints = []
    if resume:
        score, conditions = dev_score(starting_weights)
        checkpoints.append(dict(episode=512, path=str(resumed_path.relative_to(OUT)),
                                sha256=old.base.digest(resumed_path),
                                dev_score=score, dev_conditions=conditions))
        print(f'seed={seed} resumed ep512 dev_score={score:.4f}', flush=True)
    failures = []
    start_episode = 512 if resume else 0
    for index, (sample, kv, rate) in enumerate(
            all_tasks[start_episode:episodes], start_episode + 1):
        try:
            simulate(sample, kv, rate, actor)
        except RuntimeError as exc:
            failures.append(dict(sample=sample['name'], kv=kv, rate=rate,
                                 error=str(exc)))
        if index % 64 == 0:
            print(f'seed={seed} episodes={index}/{episodes} steps={actor.steps} '
                  f'updates={actor.updates} elapsed={time.monotonic()-start:.1f}s',
                  flush=True)
        if index in CHECKPOINT_EPISODES:
            path = OUT / 'checkpoints' / f'seed{seed}_ep{index}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(actor.network.state_dict(), path)
            score, conditions = dev_score(actor.network.state_dict())
            checkpoints.append(dict(episode=index, path=str(path.relative_to(OUT)),
                                    sha256=old.base.digest(path), dev_score=score,
                                    dev_conditions=conditions))
            print(f'seed={seed} checkpoint={index} dev_score={score:.4f}', flush=True)
    selected = min(checkpoints, key=lambda c: c['dev_score'])
    result = dict(seed=seed, algorithm='Double DQN', episodes=episodes,
                  train_kv=TRAIN_KV, train_rates=TRAIN_RATES,
                  train_windows_per_source=16, dev_windows_per_source=4,
                  actions=learned.ACTIONS, state_size=learned.STATE_SIZE,
                  reward='negative first-token waiting area, decode-step and queue penalties',
                  steps=actor.steps, gradient_updates=actor.updates,
                  action_counts=actor.action_counts, failures=failures,
                  checkpoints=checkpoints, selected=selected,
                  wall_s=time.monotonic()-start, gpu_training=False,
                  resumed_from_episode=512 if resume else None,
                  replay_and_optimizer_reset_at_resume=resume)
    old.base.write(OUT / 'checkpoints' / f'seed{seed}_status.json', result)
    return result


@lru_cache(maxsize=8)
def selected_weights(seed):
    status = old.base.read(OUT / 'checkpoints' / f'seed{seed}_status.json')
    path = OUT / status['selected']['path']
    assert old.base.digest(path) == status['selected']['sha256']
    return torch.load(path, map_location='cpu', weights_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'prepare'))
    parser.add_argument('--seed', type=int, default=1017)
    parser.add_argument('--episodes', type=int, default=1024)
    args = parser.parse_args()
    if args.stage == 'prepare':
        print(f'prepared {len(inputs.prepare())} windows')
    else:
        result = train(args.seed, args.episodes)
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ('checkpoints', 'action_counts')}, indent=2))
