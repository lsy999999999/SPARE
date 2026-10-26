"""Train an independent RL policy for Conversation at KV=2048."""
from __future__ import annotations

import argparse
import math
import random
import time

import torch

from research.www_rl_standalone_20261019 import policy, run


OUT = run.ROOT / 'artifacts/www_rl_standalone_conversation_20261019_r1'
RATES = (2.5, 3., 3.5, 4., 4.5)
CHECKPOINTS = (256, 512, 768, 1024)


def dev_score(weights):
    samples = [sample for sample in run.old.samples('validation')
               if sample['source'] == 'conversation' and sample['window'] >= 16]
    assert len(samples) == 4
    details = []
    for rate in (3., 3.25, 3.5, 3.75, 4.):
        rows = [run.eval_one(sample, 2048, rate, weights) for sample in samples]
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
        details.append(dict(rate_qps=rate, score=score,
                            mean_ttft_ms=mean_ttft, p99_tbt_ms=p99_tbt,
                            backlog_drift=backlog, failed_samples=failures))
    return sum(row['score'] for row in details) / len(details), details


def train(seed):
    samples = [sample for sample in run.old.samples('validation')
               if sample['source'] == 'conversation' and sample['window'] < 16]
    assert len(samples) == 16
    base_tasks = [(sample, 2048, rate) for sample in samples for rate in RATES]
    actor = policy.DQNPolicy(training=True, seed=seed)
    failures, checkpoints = [], []
    started = time.monotonic()
    for epoch in range(math.ceil(1024 / len(base_tasks))):
        tasks = base_tasks.copy()
        random.Random(seed + epoch).shuffle(tasks)
        for offset, (sample, kv, rate) in enumerate(tasks, 1):
            episode = epoch * len(base_tasks) + offset
            if episode > 1024:
                break
            try:
                run.simulate(sample, kv, rate, actor)
            except RuntimeError as exc:
                failures.append(dict(episode=episode, sample=sample['name'],
                                     kv_blocks=kv, rate_qps=rate, error=str(exc)))
            if episode % 64 == 0:
                print(f'seed={seed} episodes={episode}/1024 '
                      f'updates={actor.updates} elapsed_s={time.monotonic()-started:.1f}',
                      flush=True)
            if episode in CHECKPOINTS:
                path = OUT / 'checkpoints' / f'seed{seed}_ep{episode}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(actor.network.state_dict(), path)
                score, conditions = dev_score(actor.network.state_dict())
                checkpoints.append(dict(episode=episode,
                                        path=str(path.relative_to(OUT)),
                                        sha256=run.old.base.digest(path),
                                        dev_score=score, dev_conditions=conditions))
                print(f'seed={seed} checkpoint={episode} dev_score={score:.4f}',
                      flush=True)
    selected = min(checkpoints, key=lambda row: row['dev_score'])
    status = dict(seed=seed, algorithm='standalone Double DQN',
                  training_source='conversation', training_kv=(2048,),
                  training_rates_qps=RATES, episodes=1024,
                  train_windows=16, dev_windows=4,
                  actions=policy.ACTIONS, network_layers=(128, 128, 64),
                  state_size=policy.STATE_SIZE, discount=.9995,
                  completion_bonus=5., gradient_updates=actor.updates,
                  environment_steps=actor.steps, action_counts=actor.action_counts,
                  failures=failures, checkpoints=checkpoints, selected=selected,
                  wall_s=time.monotonic()-started, gpu_training=False)
    run.old.base.write(OUT / 'checkpoints' / f'seed{seed}_status.json', status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    print(train(args.seed)['selected'], flush=True)
