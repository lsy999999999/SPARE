"""Summarize selected checkpoints and paired 50-window capacities."""
from __future__ import annotations

import argparse
import statistics

from research.www_rl_standalone_20261019 import run


def report(kvs, seeds):
    print('Selected validation checkpoints')
    print('seed  episode  dev score  updates  failures')
    for seed in seeds:
        status = run.old.base.read(run.OUT / 'checkpoints' /
                                   f'seed{seed}_status.json')
        selected = status['selected']
        print(f'{seed}  {selected["episode"]:7d}  '
              f'{selected["dev_score"]:9.3f}  '
              f'{status["gradient_updates"]:7d}  '
              f'{len(status["failures"]):8d}')
        print('  checkpoint scores:', ', '.join(
            f'{row["episode"]}:{row["dev_score"]:.3f}'
            for row in status['checkpoints']))
    print('\nPaired capacity, mean TTFT <= 2 s and 40-60 s backlog rule')
    print('KV  source        SPARE  RL mean  RL range       delta mean  seeds')
    for kv in kvs:
        for source in ('conversation', 'code'):
            spare = run.old.base.read(run.OUT / 'groups' /
                                      f'{source}_c{kv}_spare_seed1017.json')
            values = [run.old.base.read(run.OUT / 'groups' /
                                        f'{source}_c{kv}_rl_standalone_seed{seed}.json')
                      ['capacity_qps'] for seed in seeds]
            base = spare['capacity_qps']
            mean = statistics.fmean(values)
            delta = 100 * (mean / base - 1)
            print(f'{kv:4d}  {source:12s}  {base:5.2f}  {mean:7.2f}  '
                  f'{min(values):.2f}-{max(values):.2f}  {delta:+9.2f}%  '
                  + ','.join(f'{value:.2f}' for value in values))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kvs', default='2048,4096,6144')
    parser.add_argument('--seeds', default='1017,2027,3037')
    args = parser.parse_args()
    report(tuple(map(int, args.kvs.split(','))),
           tuple(map(int, args.seeds.split(','))))
