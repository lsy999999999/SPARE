"""Run burst and cross-KV tests for the frozen Conversation-only policies."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import sys
import time

from research.www_rl_conversation_shift_20261018 import run as experiment
from research.www_rl_mixed_generalization_20261017 import burst, run as mixed, sweep


OUT = experiment.OUT
KVS = (3072, 4096, 6144, 12032)


def configure():
    mixed.OUT = OUT
    sweep.OUT = OUT
    burst.OUT = OUT / 'burst_arrival_20261018'
    for seed in experiment.SEEDS:
        status = mixed.old.base.read(OUT / 'checkpoints' / f'seed{seed}_status.json')
        assert status['train_source'] == 'conversation'
        assert status['train_kv_blocks'] == experiment.KV


def run_burst(workers):
    configure()
    burst.main(('conversation',), (.5, .2), ('rl_mixed',),
               experiment.SEEDS, workers)


def run_kv(workers):
    configure()
    tasks = [('original', 'conversation', kv, 'rl_mixed', seed)
             for kv in KVS for seed in experiment.SEEDS]
    started = time.monotonic()
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(sweep.group, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                row = future.result()
                print(f'{index}/{len(tasks)} kv={task[2]} seed={task[4]} '
                      f'capacity={row["capacity_qps"]:.3f}', flush=True)
            except Exception as exc:
                failures.append(dict(task=task, error=repr(exc)))
                print(f'FAILED {task}: {exc!r}', flush=True)
    status = dict(complete=not failures, expected=len(tasks),
                  completed=len(tasks) - len(failures), failures=failures,
                  wall_s=time.monotonic() - started)
    mixed.old.base.write(OUT / 'cross_kv_status.json', status)
    if failures:
        raise RuntimeError(status)


def pipeline(workers):
    configure()
    logs = OUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    children = []
    for stage in ('burst', 'kv'):
        log = (logs / f'{stage}.log').open('w')
        child = subprocess.Popen([sys.executable, '-m',
                                  'research.www_rl_conversation_shift_20261018.followup',
                                  stage, '--workers', str(workers)],
                                 cwd=experiment.ROOT, stdout=log,
                                 stderr=subprocess.STDOUT)
        log.close()
        children.append((stage, child))
        print(f'{stage} pid={child.pid}', flush=True)
    failed = []
    for stage, child in children:
        code = child.wait()
        print(f'{stage} exit={code}', flush=True)
        if code:
            failed.append(stage)
    if failed:
        raise RuntimeError(f'follow-up failed: {failed}; see logs')
    subprocess.run([sys.executable, '-m',
                    'research.www_rl_conversation_shift_20261018.analyze'],
                   cwd=experiment.ROOT, check=True)
    print('all follow-up comparisons complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('burst', 'kv', 'pipeline'))
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.stage == 'burst':
        run_burst(args.workers)
    elif args.stage == 'kv':
        run_kv(args.workers)
    else:
        pipeline(args.workers)
