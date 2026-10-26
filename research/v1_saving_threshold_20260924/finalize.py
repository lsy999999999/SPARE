"""Detached arithmetic audit and evidence export after the GPU pipeline exits."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_saving_threshold_20260924.common import *
from research.v1_saving_threshold_20260924.pipeline import alive


def status(state,**extra):
    write(OUT/'finalize_status.json',dict(state=state,pid=os.getpid(),time=time.time(),**extra))


def main():
    lock=(OUT/'finalize.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        status('waiting_for_pipeline')
        while True:
            pipeline=read(OUT/'pipeline_status.json')
            if alive(pipeline.get('pid',0),'v1_saving_threshold_20260924/pipeline.py'):
                time.sleep(15);continue
            break
        status('auditing',pipeline_state=pipeline['state'])
        for name in ('audit.py','analyze.py','report.py'):
            subprocess.run([sys.executable,'-B','-u',str(HERE/name)],cwd=ROOT,check=True)
        status('plotting',pipeline_state=pipeline['state'])
        subprocess.run(['/usr/bin/python3','-B',str(HERE/'figures.py')],cwd=ROOT,check=True,
            env=dict(os.environ,MPLCONFIGDIR='/tmp/query-factors-matplotlib'))
        status('finished',pipeline_state=pipeline['state'],audit=read(OUT/'audit.json')['passed'])
    except Exception as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        previous=read(OUT/'finalize_status.json') if (OUT/'finalize_status.json').exists() else {}
        if alive(previous.get('pid',0),'v1_saving_threshold_20260924/finalize.py'):
            print('Finalizer already running',previous['pid'])
        else:
            with (OUT/'finalize.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('Detached finalizer',child.pid)
    else:main()
