"""Read-only experiment status without CUDA dependencies."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import subprocess
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/v1_theory_formula_20260924_r1'
def read(p):return json.loads(p.read_text())


if __name__=='__main__':
    print(datetime.now().astimezone().isoformat(timespec='seconds'))
    for name in ('status.json','analysis/status.json'):
        if (OUT/name).exists():print(name,read(OUT/name))
    runs=[read(p) for p in (OUT/'runs').glob('*.json')]
    print('Runs:',len(runs),'completed:',sum(r['status']=='complete' for r in runs))
    print('By phase/status:',dict(Counter((r['job']['phase'],r['status']) for r in runs)))
    print('Plan: 40 development + 16 to 24 holdout + 4 to 8 A/A; objective mean TTFT only.')
    if (OUT/'manifest.json').exists():
        m=read(OUT/'manifest.json');print('GPUs:',m['gpus'],'capacity bindings:',m['bindings'])
    if (OUT/'selection.json').exists():
        f=read(OUT/'selection.json');print('Frozen winner:',f['winner'],'development scores:',f['scores'])
        print('Exact planned total:',40+m['expected_aa']+8*len(set(['old','decode',f['winner']])))
    for p in sorted(OUT.glob('worker_g*.json')):print(p.stem,read(p))
    for p in sorted(OUT.glob('calibration_g*.json')):print(p.stem,read(p))
    if (OUT/'launch.json').exists():
        pid=read(OUT/'launch.json')['pid']
        print(subprocess.run(['ps','-o','pid,ppid,sid,etime,args','-p',str(pid)],capture_output=True,text=True).stdout.strip())
