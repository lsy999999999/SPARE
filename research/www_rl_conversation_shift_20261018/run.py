"""Train on KV=2048 Conversation only, then test Conversation and held-out Code."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time

import torch

from research.www_rl_mixed_generalization_20261017 import frontier, run as mixed, sweep
from research.www_rl_mixed_generalization_20261017.policy import ACTIONS, DQNPolicy, STATE_SIZE


ROOT = mixed.old.ROOT
OUT = ROOT / 'artifacts/www_rl_conversation_shift_20261018_r1'
SEEDS = (1017, 2027, 3037)
KV = 2048
TRAIN_RATES = (.5, 2., 4., 6.)
DEV_RATES = (2., 3., 4., 5.)
EPISODES = 1024
CHECKPOINTS = (512, 1024)


def windows():
    validation = mixed.old.samples('validation')
    train = [s for s in validation if s['source'] == 'conversation' and s['window'] < 16]
    dev = [s for s in validation if s['source'] == 'conversation' and s['window'] >= 16]
    test = mixed.old.samples('test')
    assert len(train) == 16 and len(dev) == 4
    assert len([s for s in test if s['source'] == 'conversation']) == 50
    assert len([s for s in test if s['source'] == 'code']) == 50
    assert not ({s['name'] for s in train} & {s['name'] for s in dev})
    assert not ({s['name'] for s in train + dev} & {s['name'] for s in test})
    return train, dev, test


def cohort_digest(cohort):
    hashes = [mixed.old.base.digest(ROOT / s['input']) for s in cohort]
    return hashlib.sha256(''.join(hashes).encode()).hexdigest()


def protocol():
    train, dev, test = windows()
    return dict(algorithm='Double DQN', training_source='conversation',
                training_kv_blocks=KV, training_rates_qps=TRAIN_RATES,
                development_source='conversation', development_rates_qps=DEV_RATES,
                evaluation_sources=('conversation', 'code'),
                training_windows=len(train), development_windows=len(dev),
                test_windows_per_source=50, episodes=EPISODES,
                checkpoint_episodes=CHECKPOINTS, seeds=SEEDS,
                train_input_sha256=cohort_digest(train),
                dev_input_sha256=cohort_digest(dev),
                test_input_sha256={source: cohort_digest(
                    [s for s in test if s['source'] == source])
                    for source in ('conversation', 'code')},
                mean_ttft_limits_ms=frontier.LIMITS_MS,
                p99_tbt_limit_ms=200, backlog_limit='max(2, offered QPS)',
                frozen_during_evaluation=True)


def development_score(weights, dev):
    rows = []
    for rate in DEV_RATES:
        results = [mixed.eval_one(sample, KV, rate, weights) for sample in dev]
        valid = [row for row in results if row['valid']]
        failed = len(results) - len(valid)
        mean_ttft = (sum(row['mean_ttft_ms'] for row in valid) / len(valid)
                     if valid else 1e6)
        p99_tbt = (sum(row['p99_tbt_ms'] for row in valid) / len(valid)
                   if valid else 1e6)
        backlog = (sum(row['backlog_drift'] for row in valid) / len(valid)
                   if valid else 1e6)
        score = (math.log1p(mean_ttft) + 2 * failed
                 + max(0, p99_tbt - 200) / 200
                 + max(0, backlog - max(2, rate)) / max(2, rate))
        rows.append(dict(rate_qps=rate, score=score, mean_ttft_ms=mean_ttft,
                         p99_tbt_ms=p99_tbt, backlog_drift=backlog,
                         failed_samples=failed))
    return sum(row['score'] for row in rows) / len(rows), rows


def train(seed):
    torch.set_num_threads(1)
    train_samples, dev_samples, _ = windows()
    actor = DQNPolicy(training=True, seed=seed)
    base_tasks = [(sample, KV, rate) for sample in train_samples for rate in TRAIN_RATES]
    assert len(base_tasks) == 64
    started = time.monotonic()
    failures, checkpoints = [], []
    for epoch in range(EPISODES // len(base_tasks)):
        tasks = base_tasks.copy()
        random.Random(seed + epoch).shuffle(tasks)
        for offset, (sample, kv, rate) in enumerate(tasks, 1):
            episode = epoch * len(base_tasks) + offset
            try:
                mixed.simulate(sample, kv, rate, actor)
            except RuntimeError as exc:
                failures.append(dict(episode=episode, sample=sample['name'],
                                     rate_qps=rate, error=str(exc)))
            if episode % 32 == 0:
                print(f'seed={seed} episodes={episode}/{EPISODES} '
                      f'steps={actor.steps} updates={actor.updates} '
                      f'elapsed_s={time.monotonic()-started:.1f}', flush=True)
            if episode in CHECKPOINTS:
                path = OUT / 'checkpoints' / f'seed{seed}_ep{episode}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(actor.network.state_dict(), path)
                score, detail = development_score(actor.network.state_dict(), dev_samples)
                checkpoints.append(dict(episode=episode,
                                        path=str(path.relative_to(OUT)),
                                        sha256=mixed.old.base.digest(path),
                                        dev_score=score, dev_conditions=detail))
                print(f'seed={seed} checkpoint={episode} dev_score={score:.4f}',
                      flush=True)
    selected = min(checkpoints, key=lambda item: item['dev_score'])
    status = dict(seed=seed, algorithm='Double DQN', episodes=EPISODES,
                  train_source='conversation', train_kv_blocks=KV,
                  train_rates_qps=TRAIN_RATES, train_windows=16, dev_windows=4,
                  train_input_sha256=cohort_digest(train_samples),
                  dev_input_sha256=cohort_digest(dev_samples),
                  actions=ACTIONS, state_size=STATE_SIZE,
                  steps=actor.steps, gradient_updates=actor.updates,
                  action_counts=actor.action_counts, failures=failures,
                  checkpoints=checkpoints, selected=selected,
                  wall_s=time.monotonic() - started, gpu_training=False)
    mixed.old.base.write(OUT / 'checkpoints' / f'seed{seed}_status.json', status)
    print(f'seed={seed} selected={selected["episode"]} complete', flush=True)


def evaluate(workers):
    for seed in SEEDS:
        status = mixed.old.base.read(OUT / 'checkpoints' / f'seed{seed}_status.json')
        assert status['train_source'] == 'conversation' and status['train_kv_blocks'] == KV
    # The established frontier scanner uses these module globals for cache and checkpoint paths.
    mixed.OUT = OUT
    sweep.OUT = OUT
    frontier.OUT = OUT / 'frontier'
    frontier.main(('spare', 'rl_mixed'), SEEDS, workers)
    mixed.old.base.write(OUT / 'evaluation_complete.json', dict(
        complete=True, protocol=protocol(),
        internal_arm_name='rl_mixed', public_arm_name='RL-Conversation'))


def pipeline(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    mixed.old.base.write(OUT / 'protocol.json', protocol())
    logs = OUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    children = []
    for seed in SEEDS:
        log = (logs / f'train_seed{seed}.log').open('a')
        child = subprocess.Popen([sys.executable, '-m',
                                  'research.www_rl_conversation_shift_20261018.run', 'train',
                                  '--seed', str(seed)], cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT)
        log.close()
        children.append((seed, child))
        print(f'train seed={seed} pid={child.pid}', flush=True)
    failed = []
    for seed, child in children:
        code = child.wait()
        print(f'train seed={seed} exit={code}', flush=True)
        if code:
            failed.append(seed)
    if failed:
        raise RuntimeError(f'training failed for seeds {failed}; see logs')
    print('all training complete; starting paired frontier evaluation', flush=True)
    evaluate(workers)
    print('paired frontier evaluation complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('protocol', 'train', 'evaluate', 'pipeline'))
    parser.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.stage == 'protocol':
        print(json.dumps(protocol(), indent=2))
    elif args.stage == 'train':
        train(args.seed)
    elif args.stage == 'evaluate':
        evaluate(args.workers)
    else:
        pipeline(args.workers)
