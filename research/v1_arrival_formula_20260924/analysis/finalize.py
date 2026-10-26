"""Detached completion watcher: automatic independent audit, tables and figures."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_arrival_formula_20260924.common import *


def main():
    from research.v1_state_aware_20260923.run import acquire
    lock=acquire(OUT/'analysis/finalize.lock')
    write(OUT/'analysis/status.json',dict(state='waiting_for_gpu',pid=os.getpid()))
    while True:
        status=read(OUT/'status.json')
        if status['state']=='failed':raise RuntimeError(status['error'])
        if status['state']=='complete':break
        alive=subprocess.run(['pgrep','-f',str(HERE/'run.py')+'$'],capture_output=True).returncode==0
        if not alive:raise RuntimeError('GPU controller exited before completion')
        time.sleep(20)
    write(OUT/'analysis/status.json',dict(state='auditing',pid=os.getpid()))
    subprocess.run([sys.executable,'-B',str(HERE/'analysis/report.py')],cwd=ROOT,check=True)
    write(OUT/'analysis/status.json',dict(state='complete',updated=time.time()))
    lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    (OUT/'analysis').mkdir(exist_ok=True)
    if a.launch:
        with (OUT/'analysis/finalize.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('Detached finalizer',child.pid)
    else:
        try:main()
        except Exception:
            write(OUT/'analysis/status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()));raise
