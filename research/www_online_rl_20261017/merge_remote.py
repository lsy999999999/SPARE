"""Merge finished remote groups that local workers have not started."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import sys

from research.www_online_rl_20261017 import run


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(source_root):
    source = Path(source_root) / 'artifacts/www_online_rl_20261017_r1'
    status_path = source / 'remote_status.json'
    if status_path.exists():
        status = run.base.read(status_path)
        assert status['complete'] and status['groups'] == 17
        assert status['checkpoint_sha256'] == digest(run.OUT / 'checkpoint.json')
    else:
        assert digest(source / 'checkpoint.json') == digest(run.OUT / 'checkpoint.json')
        groups = list((source / 'groups').glob('conversation_c*_online_rl.json'))
        status = dict(groups=len(groups),
                      kv=[int(path.name.split('_c')[1].split('_')[0]) for path in groups],
                      group_sha256={path.name.split('_c')[1].split('_')[0]: digest(path)
                                    for path in groups})
    local_rates = run.OUT / 'rates'
    started = [int(p.name.split('_c')[1].split('_')[0])
               for p in local_rates.glob('conversation_c*_online_rl_r*.json')]
    max_started = max(started, default=0)
    allowed_min = max_started + 1024
    merged, skipped = [], []
    for kv in sorted((int(x) for x in status['kv']), reverse=True):
        remote_group = source / 'groups' / f'conversation_c{kv}_online_rl.json'
        local_group = run.OUT / 'groups' / remote_group.name
        assert digest(remote_group) == status['group_sha256'][str(kv)]
        rates = sorted((source / 'rates').glob(f'conversation_c{kv}_online_rl_r*.json'))
        assert rates
        for rate_path in rates:
            record = run.base.read(rate_path)
            assert len(record['rows']) == 50
            rate = float(rate_path.stem.split('_r')[-1])
            assert record['summary'] == run.prior.old.summary(record['rows'], rate)
        if (kv < allowed_min or local_group.exists() or
                list(local_rates.glob(f'conversation_c{kv}_online_rl_r*.json'))):
            skipped.append(kv)
            continue
        local_rates.mkdir(parents=True, exist_ok=True)
        local_group.parent.mkdir(parents=True, exist_ok=True)
        for remote_rate in rates:
            shutil.copy2(remote_rate, local_rates / remote_rate.name)
        shutil.copy2(remote_group, local_group)
        merged.append(kv)
    result = dict(remote_groups=status['groups'], local_max_started_at_merge=max_started,
                  minimum_safe_kv=allowed_min, merged=merged, skipped=skipped)
    run.base.write(run.OUT / 'merge_status.json', result)
    print(result)


if __name__ == '__main__':
    main(sys.argv[1])
