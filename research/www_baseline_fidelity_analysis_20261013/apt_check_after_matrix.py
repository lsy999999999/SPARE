"""Finish the previously timed-out numerical check after the formal GPU matrix."""
import sys
import os
import json
import time
import subprocess
import fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/www_baseline_fidelity_20261013_r1'
dest=OUT/'apt_kernel_followup';dest.mkdir(parents=True,exist_ok=True)
while True:
    status=OUT/'status.json'
    if status.exists() and json.loads(status.read_text()).get('state') in ('complete','finished_with_failures','deadline_reached','failed'):break
    time.sleep(30)
lock=open('/tmp/query-capacity-gpu0.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX)
while subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(30)
os.environ['CUDA_VISIBLE_DEVICES']='0'
sys.path.insert(0,str(ROOT))
from research.www_author_reproduction_20261007 import apt_kernel_check as check
check.OUT=dest
results=[]
try:
    for nkv,nhidden in [(2,1),(1,4),(0,3)]:
        check.case(nkv,nhidden)
        results.append(json.loads((dest/f'apt_kernel_{nkv}_{nhidden}.json').read_text()))
    (dest/'status.json').write_text(json.dumps(dict(state='complete',passed=True,cases=results,updated=time.time()),indent=2))
except BaseException:
    import traceback
    (dest/'status.json').write_text(json.dumps(dict(state='failed',passed=False,cases=results,error=traceback.format_exc(),updated=time.time()),indent=2))
    raise
