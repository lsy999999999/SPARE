"""Detached validation -> bounded GPU controller -> full offline audit."""
import os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_candidate_transfer_20260928.common import *

def main():
    verify();assert not (OUT/'launcher.json').exists()
    env=dict(os.environ,VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    code="import subprocess,sys\nfor s in ('validate.py','run.py','analyze.py'):\n subprocess.run([sys.executable,'-B','research/v1_candidate_transfer_20260928/'+s],check=True)\n"
    with (OUT/'launcher.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-B','-u','-c',code],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(pid=p.pid,started=time.time(),detached=True);write(OUT/'launcher.json',info);print(info)

if __name__=='__main__':main()
