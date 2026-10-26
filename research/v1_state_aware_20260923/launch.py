"""Start/resume with no controlling terminal, no input and persistent log files."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_state_aware_20260923.common import ROOT,HERE,OUT,write,read


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Controller already running; use status.py'); return
    if (OUT/'status.json').exists() and read(OUT/'status.json').get('state')=='finished':
        print('Already complete:',OUT/'RESULTS_CN.md'); return
    env=dict(os.environ,VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with (OUT/'controller.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'run.py')],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    write(OUT/'launch.json',dict(pid=child.pid,detached=True,command='run.py',output=str(OUT)))
    print(f'Detached controller PID: {child.pid}\nOutput: {OUT}',flush=True)


if __name__=='__main__': main()
