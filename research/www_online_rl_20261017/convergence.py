"""Three-seed held-out learning curves for the fixed Q-learning protocol."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import random
import statistics
import time

from research.www_online_rl_20261017 import run

SEEDS = (1017, 2027, 3037)
CHECKPOINTS = (0, 128, 256, 384, 512)
VALIDATION_KV = (1024, 4096)
VALIDATION_RATES = (.5, 4.)


def evaluate(policy, heldout):
    rows = []
    for kv in VALIDATION_KV:
        for rate in VALIDATION_RATES:
            for sample in heldout:
                result = run.simulate(sample, kv, rate, policy)
                rows.append(dict(kv_blocks=kv, rate=rate, source=sample['source'],
                                 name=sample['name'],
                                 mean_ttft_ms=run.base.probe(result)['mean_ttft_ms']))
    summary = []
    for kv in VALIDATION_KV:
        for rate in VALIDATION_RATES:
            cohort = [r['mean_ttft_ms'] for r in rows
                      if r['kv_blocks'] == kv and r['rate'] == rate]
            summary.append(dict(kv_blocks=kv, rate=rate, windows=len(cohort),
                                mean_ttft_ms=statistics.fmean(cohort)))
    return summary


def one_seed(seed):
    import torch
    torch.set_num_threads(1)
    policy = run.QPolicy(training=True, seed=seed)
    training = [s for s in run.samples('validation') if s['window'] < 16]
    heldout = [s for s in run.samples('validation') if s['window'] >= 16]
    assert len(training) == 32 and len(heldout) == 8
    tasks = [(s, kv, rate) for kv in run.TRAIN_KV for rate in run.TRAIN_RATES
             for s in training]
    random.Random(seed).shuffle(tasks)
    started, cpu_started = time.monotonic(), time.process_time()
    points = []
    for index in range(len(tasks)+1):
        if index in CHECKPOINTS:
            frozen = run.QPolicy(table=policy.dump())
            points.append(dict(training_episodes=index, td_updates=policy.updates,
                               summary=evaluate(frozen, heldout)))
        if index < len(tasks):
            sample, kv, rate = tasks[index]
            run.simulate(sample, kv, rate, policy)
    table = policy.dump()
    table_sha = hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest()
    original = run.base.read(run.OUT / 'checkpoint.json')
    result = dict(seed=seed, train_episodes=len(tasks),
                  heldout_windows_per_source=4, points=points,
                  final_q_table_sha256=table_sha,
                  matches_original_checkpoint=(table == original['q_table']) if seed == 1017 else None,
                  wall_s=time.monotonic()-started,
                  cpu_s=time.process_time()-cpu_started)
    run.base.write(run.OUT / f'convergence_seed{seed}.json', result)
    return dict(seed=seed, wall_s=result['wall_s'],
                matches_original_checkpoint=result['matches_original_checkpoint'])


def main():
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        for future in as_completed([pool.submit(one_seed, seed) for seed in SEEDS]):
            print(future.result(), flush=True)
    run.base.write(run.OUT / 'convergence_status.json',
                   dict(complete=True, seeds=SEEDS, checkpoints=CHECKPOINTS))


if __name__ == '__main__':
    main()
