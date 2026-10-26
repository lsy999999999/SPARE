"""Small paired development diagnostic; never used as a test result."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics

import torch

from research.www_kv_extreme_stress_20261016 import run as sim
from research.www_rl_standalone_20261019 import run


def main(seed, episode, source, kv, rate, out_dir):
    samples = [sample for sample in run.old.samples('validation')
               if sample['window'] >= 16 and sample['source'] == source]
    assert len(samples) == 4
    weights = torch.load(out_dir / 'checkpoints' /
                         f'seed{seed}_ep{episode}.pt', map_location='cpu',
                         weights_only=True)
    if not hasattr(run.old.base, 'SIM'):
        sim.initialize()
    for arm in ('spare', 'rl_standalone'):
        rows = [(sim.simulate_sample(sample, kv, 'arrival', rate)
                 if arm == 'spare' else run.eval_one(sample, kv, rate, weights))
                for sample in samples]
        valid = [row for row in rows if row['valid']]
        print(f'{arm}: valid={len(valid)}/4 '
              f'mean_TTFT_ms={statistics.fmean(row["mean_ttft_ms"] for row in valid):.1f} '
              f'P99_TBT_ms={statistics.fmean(row["p99_tbt_ms"] for row in valid):.1f} '
              f'backlog_drift={statistics.fmean(row["backlog_drift"] for row in valid):.2f}',
              flush=True)
        if arm == 'rl_standalone':
            counts = [sum(row['rl_actions'][index] for row in valid)
                      for index in range(len(run.learned.ACTIONS))]
            print('actions:', sorted(zip(run.learned.ACTIONS, counts),
                                     key=lambda pair: -pair[1]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--source', choices=('code', 'conversation'), required=True)
    parser.add_argument('--kv', type=int, default=2048)
    parser.add_argument('--rate', type=float, default=3.5)
    parser.add_argument('--out-dir', type=Path, default=run.OUT)
    args = parser.parse_args()
    main(args.seed, args.episode, args.source, args.kv, args.rate,
         args.out_dir.resolve())
