"""Detached controller plus mandatory independent report; all failures are persisted."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_decode_arrival_confirm_20260925.common import *

GPU_PYTHON='.venv/bin/python'


def main():
    from research.v1_state_aware_20260923.run import acquire
    lock=acquire(OUT/'analysis/pipeline.lock')
    write(OUT/'analysis/status.json',dict(state='running_gpu',pid=os.getpid(),updated=time.time()))
    with (OUT/'controller.log').open('a') as log:
        subprocess.run([GPU_PYTHON,'-B','-u',str(HERE/'run.py')],cwd=ROOT,
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(OUT/'analysis/status.json',dict(state='auditing',pid=os.getpid(),updated=time.time()))
    subprocess.run([sys.executable,'-B',str(HERE/'analysis/report.py')],cwd=ROOT,check=True)
    write(OUT/'analysis/status.json',dict(state='complete',updated=time.time()))
    lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');a=parser.parse_args()
    (OUT/'analysis').mkdir(exist_ok=True)
    if a.launch:
        with (OUT/'analysis/pipeline.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],cwd=ROOT,
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(OUT/'launch.json',dict(pid=child.pid,time=time.time(),detached=True))
        print('Detached pipeline',child.pid)
    else:
        try:main()
        except Exception:
            write(OUT/'analysis/status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()));raise
