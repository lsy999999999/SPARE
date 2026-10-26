"""Train one seed, then run held-out capacity scans without supervision."""
from __future__ import annotations

import argparse
import time

from research.www_rl_standalone_20261019 import run
from research.www_rl_standalone_v2_20261019 import sweep, train


CELLS = (('conversation', 2048), ('code', 2048),
         ('conversation', 4096), ('code', 4096))


def main(seed, episodes):
    try:
        status = train.train(seed, episodes)
        capacities = []
        for source, kv in CELLS:
            for arm in ('spare', 'rl_standalone'):
                progress = dict(state='testing', seed=seed, source=source,
                                kv_blocks=kv, arm=arm,
                                selected_episode=status['selected']['episode'],
                                completed_cells=capacities)
                run.old.base.write(train.OUT / f'seed{seed}_progress.json', progress)
                result = sweep.scan(source, kv, arm, seed)
                capacities.append(dict(source=source, kv_blocks=kv, arm=arm,
                                       capacity_qps=result['capacity_qps']))
        run.old.base.write(train.OUT / f'seed{seed}_progress.json', dict(
            state='complete', seed=seed, selected_episode=status['selected']['episode'],
            completed_cells=capacities, completed_at=time.time()))
    except Exception as exc:
        run.old.base.write(train.OUT / f'seed{seed}_progress.json', dict(
            state='error', seed=seed, error=repr(exc), failed_at=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=train.SEEDS, required=True)
    parser.add_argument('--episodes', type=int, choices=train.CHECKPOINTS,
                        default=1536)
    args = parser.parse_args()
    main(args.seed, args.episodes)
