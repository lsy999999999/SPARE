"""One-time, outcome-independent whole-group migration away from unstable GPU 0."""
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.v1_arrival_formula_20260924.common import OUT, read, write, digest
from research.v1_state_aware_20260923 import run as infra


def main():
    lock = infra.acquire(OUT / 'controller.lock')
    m = read(OUT / 'manifest.json')
    assert m['bindings'] == {'768_code': 0, '768_conversation': 6, '2048_code': 7, '2048_conversation': 0}
    assert read(OUT / 'status.json')['state'] == 'failed'
    hardware = infra.hardware()
    assert all(infra.idle(hardware[g]) for g in (6, 7))
    archive = OUT / 'recovery' / ('rebind_' + str(time.time_ns()))
    archive.mkdir(parents=True)
    for name in ('manifest.json', 'status.json', 'controller.log', 'analysis/status.json', 'worker_g0.json'):
        source = OUT / name
        if source.exists():
            target = archive / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    moved = []
    for path in sorted((OUT / 'runs').glob('*.json')):
        r = read(path)
        if r['job']['phase'] != 'compare' or r['gpu'] != 0:
            continue
        for source in (path, OUT / r['rounds_path']):
            target = archive / source.relative_to(OUT)
            target.parent.mkdir(parents=True, exist_ok=True)
            moved.append(dict(path=str(source.relative_to(OUT)), sha256=digest(source)))
            shutil.move(str(source), str(target))
    assert len(moved) == 14, 'Expected six old GPU0 results plus one result from interrupted task'
    for source in sorted((OUT / 'contamination').glob('*.json')):
        target = archive / 'contamination' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        moved.append(dict(path=str(source.relative_to(OUT)), sha256=digest(source)))
        shutil.move(str(source), str(target))
    old = dict(m['bindings'])
    m['bindings'].update({'768_code': 6, '2048_conversation': 7})
    m['active_gpus'] = [6, 7]
    m['resource_amendment'] = dict(time=time.time(), archive=str(archive.relative_to(OUT)),
                                 original_bindings=old, reason='Repeated external activity on GPU0; rerun entire groups on idle GPUs')
    write(archive / 'migration.json', dict(original_bindings=old, new_bindings=m['bindings'],
                                        archived_files=moved, original_calibrations_retained=True))
    write(OUT / 'manifest.json', m)
    write(OUT / 'worker_g0.json', dict(state='retired_after_external_activity', updated=time.time()))
    print('Preserved', len(moved), 'files in', archive)
    print('Bindings:', m['bindings'])
    lock.close()


if __name__ == '__main__':
    main()
