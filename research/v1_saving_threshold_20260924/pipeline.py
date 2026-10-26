"""Persistent supervisor: main scan -> branches -> fit -> formula GPU check."""
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


def alive(pid,needle):
    try:return needle.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except (FileNotFoundError,ProcessLookupError):return False


def state(name,**kw):
    write(OUT/'pipeline_status.json',dict(state=name,pid=os.getpid(),time=time.time(),**kw))


def call(script,*args):
    subprocess.run([sys.executable,'-B','-u',str(HERE/script),*args],cwd=ROOT,check=True)


def main():
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        state('waiting_for_main')
        while True:
            s=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
            if alive(s.get('pid',0),'v1_saving_threshold_20260924/run.py'):
                time.sleep(10);continue
            break
        if s.get('stage')=='main' and s.get('missing') and time.time()<DEADLINE:
            state('resuming_main');call('run.py','--controller','--stage','main')
        if time.time()<DEADLINE:
            state('running_branches');call('run.py','--controller','--stage','extra')
        state('fitting')
        call('analyze.py','--fit')
        if time.time()<DEADLINE:
            state('running_formula_check');call('run.py','--controller','--stage','extra')
        state('summarizing');call('analyze.py')
        state('finished')
    except Exception as exc:
        state('failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        old=read(OUT/'pipeline_status.json') if (OUT/'pipeline_status.json').exists() else {}
        if alive(old.get('pid',0),'v1_saving_threshold_20260924/pipeline.py'):
            print('Pipeline already running',old['pid'])
        else:
            with (OUT/'pipeline.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('Detached full pipeline',child.pid)
    else:main()
