"""Train a standalone Double DQN on the paired WWW simulator windows."""
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
from research.www_rl_standalone_20261019 import policy as learned


ROOT = old.ROOT
OUT = ROOT / 'artifacts/www_rl_standalone_20261019_r2'
TRAIN_KV = (1024, 2048, 4096, 8192)
TRAIN_RATES = (.5, 2., 4., 6.)
CHECKPOINT_EPISODES = (256, 512, 1024, 1536, 2048)
SEEDS = (1017, 2027, 3037)


def independent_select(rows, budget, chunk, free, total, block_size,
                       concurrency, max_model_len, *, options, **kwargs):
    assert options['mode'] == 'fixed'
    return learned.select_action(rows, budget, chunk, free, total, block_size,
                                 concurrency, max_model_len,
                                 action=options['standalone_action'])


def make_scheduler(policy):
    from research.www_evaluation_completion_20261003.scheduler import EvaluationScheduler

    class IndependentScheduler(EvaluationScheduler):
        def schedule(self):
            state = getattr(policy, 'state_of', learned.state_of)(self)
            if self.round_rows and hasattr(self, '_rl_previous') and policy.training:
                previous = self.round_rows[-1]
                reward = getattr(policy, 'reward_of', learned.reward_of)(previous)
                policy.observe(*self._rl_previous, reward, state)
            action = policy.act(state)
            self.evaluation_options = {'mode': 'fixed', 'standalone_action': action}
            result = super().schedule()
            self.round_rows[-1]['rl_pending_first_tokens'] = sum(
                not r.output_token_ids for r in self.requests.values())
            self.round_rows[-1]['rl_decode_count'] = sum(
                bool(r.output_token_ids) for r in self.running)
            now = self.arrival_clock()
            self.round_rows[-1]['rl_pending_ages_s'] = [
                max(0., now - r.arrival_time)
                for r in self.requests.values() if not r.output_token_ids]
            self.round_rows[-1]['rl_action'] = action
            self._rl_previous = (state, action)
            return result

    return IndependentScheduler


def simulate(sample, kv, rate, policy):
    from research.www_evaluation_completion_20261003 import scheduler as evaluation

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
    selector = getattr(policy, 'select_action', independent_select)
    with patch.object(sim, 'make_scheduler', make), \
         patch.object(evaluation, 'select', selector):
        result = sim.simulate(requests, arm='arrival', capacity=kv, budget=1024,
                              chunk=256, history=old.base.HISTORY[sample['source']],
                              seed=seed, fitted_timing=Clock(), capture_probe=True,
                              max_wall_s=60)
    scheduler = capture['scheduler']
    if policy.training and hasattr(scheduler, '_rl_previous'):
        last = scheduler.round_rows[-1]
        reward = getattr(policy, 'reward_of', learned.reward_of)(last)
        state = getattr(policy, 'state_of', learned.state_of)(scheduler)
        policy.observe(*scheduler._rl_previous, reward, state, done=True)
    return result


def eval_one(sample, kv, rate, weights):
    actor = learned.DQNPolicy(weights=weights)
    started = time.monotonic()
    try:
        result = simulate(sample, kv, rate, actor)
        row = dict(valid=True, **old.base.probe(result),
                   schedule_mean_ms=result['metrics']['cpu_schedule_including_audit_ms_mean'],
                   rl_actions=actor.action_counts,
                   rounds=len(result['probe_records']))
    except RuntimeError as exc:
        if str(exc) not in ('empty schedule with pending work; trace/policy cannot progress',
                            'simulation limit; no completed result will be reported'):
            raise
        row = dict(valid=False, failure=str(exc),
                   requests=len(old.base.generate(sample, rate)))
    return dict(source=sample['source'], name=sample['name'], day=sample['day'],
                capacity=kv, arm='rl_standalone', budget=1024, rate=rate,
                wall_s=time.monotonic() - started, **row)


def development_score(weights):
    heldout = [s for s in old.samples('validation') if s['window'] >= 16]
    assert len(heldout) == 8
    conditions = ([(2048, rate) for rate in (3., 3.5, 4.)]
                  + [(4096, rate) for rate in (3.5, 4.5, 5.)])
    results = []
    for kv, rate in conditions:
        rows = [eval_one(sample, kv, rate, weights) for sample in heldout]
        valid = [row for row in rows if row['valid']]
        failures = len(rows) - len(valid)
        mean_ttft = (sum(row['mean_ttft_ms'] for row in valid) / len(valid)
                     if valid else 1e6)
        p99_tbt = (sum(row['p99_tbt_ms'] for row in valid) / len(valid)
                   if valid else 1e6)
        backlog = (sum(row['backlog_drift'] for row in valid) / len(valid)
                   if valid else 1e6)
        score = (math.log1p(mean_ttft) + 2 * failures
                 + 4 * max(0, mean_ttft - 2000) / 2000
                 + max(0, p99_tbt - 200) / 200
                 + max(0, backlog - max(2, rate)) / max(2, rate))
        results.append(dict(kv_blocks=kv, rate_qps=rate, score=score,
                            mean_ttft_ms=mean_ttft, p99_tbt_ms=p99_tbt,
                            backlog_drift=backlog, failed_samples=failures))
    return sum(row['score'] for row in results) / len(results), results


def train(seed, episodes=2048):
    assert seed in SEEDS and episodes in CHECKPOINT_EPISODES
    chosen = [s for s in old.samples('validation') if s['window'] < 16]
    assert len(chosen) == 32
    base_tasks = [(sample, kv, rate) for sample in chosen
                  for kv in TRAIN_KV for rate in TRAIN_RATES]
    assert len(base_tasks) == 512
    actor = learned.DQNPolicy(training=True, seed=seed)
    started = time.monotonic()
    checkpoints, failures = [], []
    for epoch in range(math.ceil(episodes / len(base_tasks))):
        tasks = base_tasks.copy()
        random.Random(seed + epoch).shuffle(tasks)
        for offset, (sample, kv, rate) in enumerate(tasks, 1):
            episode = epoch * len(base_tasks) + offset
            if episode > episodes:
                break
            try:
                simulate(sample, kv, rate, actor)
            except RuntimeError as exc:
                failures.append(dict(episode=episode, sample=sample['name'],
                                     kv_blocks=kv, rate_qps=rate, error=str(exc)))
            if episode % 64 == 0:
                print(f'seed={seed} episodes={episode}/{episodes} '
                      f'updates={actor.updates} elapsed_s={time.monotonic()-started:.1f}',
                      flush=True)
            if episode in CHECKPOINT_EPISODES:
                path = OUT / 'checkpoints' / f'seed{seed}_ep{episode}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(actor.network.state_dict(), path)
                score, conditions = development_score(actor.network.state_dict())
                checkpoints.append(dict(episode=episode,
                                        path=str(path.relative_to(OUT)),
                                        sha256=old.base.digest(path), dev_score=score,
                                        dev_conditions=conditions))
                print(f'seed={seed} checkpoint={episode} dev_score={score:.4f}',
                      flush=True)
        if (epoch + 1) * len(base_tasks) >= episodes:
            break
    selected = min(checkpoints, key=lambda item: item['dev_score'])
    status = dict(seed=seed, algorithm='standalone Double DQN', episodes=episodes,
                  training_kv=TRAIN_KV, training_rates_qps=TRAIN_RATES,
                  train_windows_per_source=16, dev_windows_per_source=4,
                  actions=learned.ACTIONS, state_size=learned.STATE_SIZE,
                  network_layers=(128, 128, 64), gradient_updates=actor.updates,
                  discount=.9995, completion_bonus=5., update_every_steps=32,
                  environment_steps=actor.steps, action_counts=actor.action_counts,
                  failures=failures, checkpoints=checkpoints, selected=selected,
                  wall_s=time.monotonic() - started, gpu_training=False)
    old.base.write(OUT / 'checkpoints' / f'seed{seed}_status.json', status)
    return status


@lru_cache(maxsize=8)
def selected_weights(seed):
    status = old.base.read(OUT / 'checkpoints' / f'seed{seed}_status.json')
    path = OUT / status['selected']['path']
    assert old.base.digest(path) == status['selected']['sha256']
    return torch.load(path, map_location='cpu', weights_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'smoke'))
    parser.add_argument('--seed', type=int, choices=SEEDS, default=1017)
    parser.add_argument('--episodes', type=int, choices=CHECKPOINT_EPISODES, default=2048)
    args = parser.parse_args()
    if args.stage == 'smoke':
        sample = next(s for s in old.samples('validation') if s['source'] == 'conversation')
        actor = learned.DQNPolicy(training=True, seed=args.seed)
        result = simulate(sample, 2048, 3., actor)
        print(json.dumps(dict(rounds=len(result['probe_records']),
                              actions=actor.action_counts,
                              mean_ttft_ms=old.base.probe(result)['mean_ttft_ms'])))
    else:
        print(json.dumps(train(args.seed, args.episodes)['selected'], indent=2))
