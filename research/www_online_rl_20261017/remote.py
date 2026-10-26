"""Run disjoint high-KV Conversation groups on the 4090 host CPUs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time

from research.www_online_rl_20261017 import run


def main():
    tasks = [('conversation', kv) for kv in run.EVAL_KV if kv >= 4096]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        for index, future in enumerate(as_completed([pool.submit(run.group, task)
                                                     for task in tasks]), 1):
            result = future.result()
            print(f'{index}/{len(tasks)} KV={result["kv_blocks"]} '
                  f'capacity={result["capacity_qps"]:.2f}', flush=True)
    checks = {}
    for _, kv in tasks:
        path = run.OUT / 'groups' / f'conversation_c{kv}_online_rl.json'
        checks[str(kv)] = hashlib.sha256(path.read_bytes()).hexdigest()
    run.base.write(run.OUT / 'remote_status.json', dict(
        complete=True, host='4090', groups=len(tasks), kv=list(checks),
        group_sha256=checks, checkpoint_sha256=run.base.digest(run.OUT / 'checkpoint.json'),
        wall_s=time.monotonic()-started))
    print(json.dumps({'complete': True, 'groups': len(tasks),
                      'wall_s': time.monotonic()-started}), flush=True)


if __name__ == '__main__':
    main()
