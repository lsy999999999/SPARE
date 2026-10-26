"""Detach the controller from the invoking shell's process session."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/www_biobjective_ablation_20261016_r1'
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / 'controller.lock').open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('An ablation controller is already running; use progress.sh.')
with (OUT / 'controller.log').open('ab', buffering=0) as stream:
    process = subprocess.Popen(
        [str(ROOT / '.vllm-venv/bin/python'), '-B', '-u', '-m',
         'research.www_biobjective_ablation_20261016.run', 'run', '--workers', '12'],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True, env=dict(os.environ))
print(f'Started detached controller PID {process.pid}', flush=True)
