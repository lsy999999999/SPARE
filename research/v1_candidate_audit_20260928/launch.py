"""Start a detached, bounded validation -> controller -> independent audit."""
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_candidate_audit_20260928.common import *


def main():
    from research.v1_candidate_audit_20260928.run import verify
    verify();assert not (OUT/'launcher.json').exists()
    env=dict(os.environ,VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    code="""import subprocess,sys
for script in ('validate.py','run.py','analyze.py'):
    subprocess.run([sys.executable,'-B','research/v1_candidate_audit_20260928/'+script],check=True)
"""
    with (OUT/'launcher.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,'-B','-u','-c',code],cwd=ROOT,env=env,
             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(pid=proc.pid,started=time.time(),detached=True,log=str(OUT/'launcher.log'))
    write(OUT/'launcher.json',info);print(json.dumps(info))


if __name__=='__main__':main()
