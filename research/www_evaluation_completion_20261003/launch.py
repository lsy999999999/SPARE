"""Detach the frozen CPU controller from the interactive shell."""
import os
import subprocess
import time
from .common import ROOT, OUT, write

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
             MPLCONFIGDIR=str(OUT/'matplotlib_cache'))
    with (OUT/'controller.log').open('a') as log:
        p=subprocess.Popen([str(ROOT/'.vllm-venv/bin/python'),'-B','-m',
                            'research.www_evaluation_completion_20261003.run','--jobs','12'],
                           cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True)
    write(OUT/'launch.json',dict(pid=p.pid,started=time.time(),jobs=12,host='local',
                               command='research.www_evaluation_completion_20261003.run'))
    time.sleep(2)
    if p.poll() is not None:raise RuntimeError('Controller exited; inspect controller.log')
    print('Detached controller PID:',p.pid)

if __name__=='__main__':main()
