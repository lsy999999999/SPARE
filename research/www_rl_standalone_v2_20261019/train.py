"""Train an independent request-aware RL scheduler on Conversation, KV=2048."""
from __future__ import annotations

import argparse
import math
import random
import time

import torch

from research.www_rl_standalone_20261019 import run
from research.www_rl_standalone_v2_20261019 import policy


OUT = run.ROOT / 'artifacts/www_rl_standalone_v2_20261019_r1'
RATES = (2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5)
CHECKPOINTS = (256, 512, 768, 1024, 1280, 1536)
SEEDS = (7047, 8057, 9067)


def eval_one(sample, kv, rate, weights):
    actor = policy.DQNPolicy(weights=weights)
    try:
        result = run.simulate(sample, kv, rate, actor)
        return dict(valid=True, **run.old.base.probe(result),
                    schedule_mean_ms=result['metrics']['cpu_schedule_including_audit_ms_mean'],
                    rl_actions=actor.action_counts,
                    rounds=len(result['probe_records']))
    except RuntimeError as exc:
        if str(exc) not in ('empty schedule with pending work; trace/policy cannot progress',
                            'simulation limit; no completed result will be reported'):
            raise
        return dict(valid=False, failure=str(exc))


def dev_score(weights):
    samples = [sample for sample in run.old.samples('validation')
               if sample['source'] == 'conversation' and sample['window'] >= 16]
    assert len(samples) == 4
    conditions = []
    for rate in (3., 3.25, 3.5, 3.75, 4., 4.25):
        rows = [eval_one(sample, 2048, rate, weights) for sample in samples]
        valid = [row for row in rows if row['valid']]
        failures = len(rows) - len(valid)
        mean_ttft = (sum(row['mean_ttft_ms'] for row in valid) / len(valid)
                     if valid else 1e6)
        p99_tbt = (sum(row['p99_tbt_ms'] for row in valid) / len(valid)
                   if valid else 1e6)
        backlog = (sum(row['backlog_drift'] for row in valid) / len(valid)
                   if valid else 1e6)
        score = (mean_ttft / 2000 + 3 * max(0, mean_ttft - 2000) / 2000
                 + max(0, p99_tbt - 200) / 200
                 + max(0, backlog - max(2, rate)) / max(2, rate)
                 + 2 * failures)
        conditions.append(dict(rate_qps=rate, score=score,
                               mean_ttft_ms=mean_ttft, p99_tbt_ms=p99_tbt,
                               backlog_drift=backlog, failed_samples=failures))
    return sum(row['score'] for row in conditions) / len(conditions), conditions


def train(seed, episodes=1536):
    assert seed in SEEDS and episodes in CHECKPOINTS
    samples = [sample for sample in run.old.samples('validation')
               if sample['source'] == 'conversation' and sample['window'] < 16]
    assert len(samples) == 16
    tasks = [(sample, 2048, rate) for sample in samples for rate in RATES]
    actor = policy.DQNPolicy(training=True, seed=seed)
    failures, checkpoints = [], []
    started = time.monotonic()
    for epoch in range(math.ceil(episodes / len(tasks))):
        shuffled = tasks.copy()
        random.Random(seed + epoch).shuffle(shuffled)
        for offset, (sample, kv, rate) in enumerate(shuffled, 1):
            episode = epoch * len(tasks) + offset
            if episode > episodes:
                break
            try:
                run.simulate(sample, kv, rate, actor)
            except RuntimeError as exc:
                failures.append(dict(episode=episode, sample=sample['name'],
                                     rate_qps=rate, error=str(exc)))
            if episode % 64 == 0:
                progress = dict(state='training', seed=seed, episode=episode,
                                total_episodes=episodes, updates=actor.updates,
                                elapsed_s=time.monotonic() - started,
                                failures=len(failures))
                run.old.base.write(OUT / f'seed{seed}_progress.json', progress)
                print(f'seed={seed} episodes={episode}/{episodes} '
                      f'updates={actor.updates} elapsed_s={progress["elapsed_s"]:.1f}',
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
    status = dict(state='trained', seed=seed, algorithm='independent Double DQN v2',
                  training_source='conversation', training_kv=(2048,),
                  training_rates_qps=RATES, episodes=episodes, train_windows=16,
                  dev_windows=4, actions=policy.ACTIONS,
                  network_layers=(256, 256, 128), state_size=policy.STATE_SIZE,
                  gradient_updates=actor.updates, environment_steps=actor.steps,
                  failures=failures, checkpoints=checkpoints, selected=selected,
                  wall_s=time.monotonic() - started, gpu_training=False)
    run.old.base.write(OUT / 'checkpoints' / f'seed{seed}_status.json', status)
    run.old.base.write(OUT / f'seed{seed}_progress.json', dict(
        state='trained', seed=seed, episode=episodes, total_episodes=episodes,
        selected_episode=selected['episode'], dev_score=selected['dev_score'],
        elapsed_s=status['wall_s']))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--episodes', type=int, choices=CHECKPOINTS, default=1536)
    args = parser.parse_args()
    print(train(args.seed, args.episodes)['selected'], flush=True)
