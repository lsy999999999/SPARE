"""Launch the complete matrix detached; no SSH session is needed afterward."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/v1_capacity_gpu_20260923_r1'
OUT.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',
         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (OUT/'controller.log').open('a') as log:
    child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).with_name('run.py'))],
        cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(f'Controller PID: {child.pid}\nOutput: {OUT}',flush=True)
