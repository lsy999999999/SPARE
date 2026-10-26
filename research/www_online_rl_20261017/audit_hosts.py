"""Compare deterministic outcome fields for overlapping local and remote groups."""
from __future__ import annotations

from pathlib import Path
import sys

from research.www_online_rl_20261017 import run


FIELDS = ('valid', 'mean_ttft_ms', 'p95_ttft_ms', 'p99_tbt_ms',
          'backlog_drift', 'trajectory_sha256')


def main(root):
    remote = Path(root) / 'artifacts/www_online_rl_20261017_r1'
    status = run.base.read(remote / 'remote_status.json')
    assert status['complete'] and status['groups'] == 17
    checked, differences, pending = [], [], []
    for kv in map(int, status['kv']):
        local_group = run.OUT / 'groups' / f'conversation_c{kv}_online_rl.json'
        remote_group = remote / 'groups' / local_group.name
        if not local_group.exists():
            pending.append(kv)
            continue
        local = run.base.read(local_group)
        other = run.base.read(remote_group)
        for field in ('capacity_qps', 'next_failed_qps'):
            if local[field] != other[field]:
                differences.append(dict(kv=kv, field=field,
                                        local=local[field], remote=other[field]))
        common = {p.name for p in (remote / 'rates').glob(f'conversation_c{kv}_online_rl_r*.json')}
        common &= {p.name for p in (run.OUT / 'rates').glob(f'conversation_c{kv}_online_rl_r*.json')}
        for name in common:
            left = run.base.read(run.OUT / 'rates' / name)
            right = run.base.read(remote / 'rates' / name)
            lrows = {r['name']: r for r in left['rows']}
            rrows = {r['name']: r for r in right['rows']}
            assert set(lrows) == set(rrows) and len(lrows) == 50
            for sample in lrows:
                for field in FIELDS:
                    if lrows[sample].get(field) != rrows[sample].get(field):
                        differences.append(dict(kv=kv, rate=name, sample=sample,
                                                field=field,
                                                local=lrows[sample].get(field),
                                                remote=rrows[sample].get(field)))
        checked.append(dict(kv=kv, rates=len(common)))
    result = dict(remote_groups=status['groups'], checked=checked,
                  pending_local=pending, differences=differences)
    run.base.write(run.OUT / 'cross_host_audit.json', result)
    print(f'checked={len(checked)} pending={len(pending)} differences={len(differences)}')
    if differences:
        print(differences[:5])


if __name__ == '__main__':
    main(sys.argv[1])
