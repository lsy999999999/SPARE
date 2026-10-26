"""Detached resource-only recovery; preserve the frozen matrix and GPU bindings."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.v1_arrival_formula_20260924.common import ROOT, HERE, OUT, read, write

GPU_PYTHON = '.venv/bin/python'


def main():
    from research.v1_state_aware_20260923.run import acquire
    lock = acquire(OUT / 'analysis/recover.lock')
    history = OUT / 'recovery' / str(time.time_ns())
    history.mkdir(parents=True)
    for name in ('status.json', 'manifest.json', 'controller.log', 'analysis/status.json'):
        source = OUT / name
        if source.exists():
            target = history / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    write(history / 'reason.json', dict(
        reason='Resume current manifest and retained runs; prior resource amendments are recorded separately.',
        pid=os.getpid(), time=time.time(), policy_changed=False, bindings_changed_by_supervisor=False))
    attempt = 0
    while True:
        attempt += 1
        write(OUT / 'analysis/status.json', dict(state='recovering_gpu', pid=os.getpid(), attempt=attempt))
        with (OUT / 'controller.log').open('a') as log:
            result = subprocess.run([GPU_PYTHON, '-B', '-u', str(HERE / 'run.py')],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        status = read(OUT / 'status.json')
        write(history / f'attempt_{attempt}.json', dict(returncode=result.returncode, status=status))
        if result.returncode == 0 and status['state'] == 'complete':
            break
        error = status.get('error', '')
        if not (status['state'] == 'failed' and 'now busy; preserve pairing and resume later' in error
                and not list((OUT / 'contamination').glob('*.json'))):
            raise RuntimeError(f'Non-resource failure: {error}')
        write(OUT / 'analysis/status.json', dict(state='waiting_for_bound_gpu', pid=os.getpid(), attempt=attempt))
        time.sleep(30)
    write(OUT / 'analysis/status.json', dict(state='auditing', pid=os.getpid()))
    subprocess.run([sys.executable, '-B', str(HERE / 'analysis/report.py')], cwd=ROOT, check=True)
    write(OUT / 'analysis/status.json', dict(state='complete', updated=time.time(), recovery_history=str(history)))
    lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        with (OUT / 'analysis/recover.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve())],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print('Detached recovery supervisor', child.pid)
    else:
        try:
            main()
        except Exception:
            write(OUT / 'analysis/status.json', dict(state='failed', error=traceback.format_exc(), updated=time.time()))
            raise
