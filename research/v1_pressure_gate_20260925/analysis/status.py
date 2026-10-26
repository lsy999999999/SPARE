"""Read-only status, no CUDA or model imports."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import subprocess
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'artifacts/v1_pressure_gate_20260925_r1'
def read(p):return json.loads(p.read_text())

if __name__=='__main__':
    print(datetime.now().astimezone().isoformat(timespec='seconds'))
    for name in ('status.json','analysis/status.json'):
        if (OUT/name).exists():print(name,read(OUT/name))
    rs=[read(p) for p in (OUT/'runs').glob('*.json')]
    print('Completed',sum(r['status']=='complete' for r in rs),'of',len(rs),'saved')
    print('By phase/status:',dict(Counter((r['job']['phase'],r['status']) for r in rs)))
    if (OUT/'manifest.json').exists():
        m=read(OUT/'manifest.json');d=read(OUT/'design.json');aa={(g,int(key.split('_')[0])) for key,g in m['bindings'].items()};print('Plan',d['expected_formal_runs']+2*len(aa),'total; GPUs',m['gpus'],'bindings',m['bindings'])
    for p in sorted(OUT.glob('worker_g*.json')):print(p.stem,read(p))
    for p in sorted(OUT.glob('calibration_g*.json')):print(p.stem,read(p))
    if (OUT/'launch.json').exists():
        pid=read(OUT/'launch.json')['pid']
        print(subprocess.run(['ps','-o','pid,ppid,sid,etime,args','-p',str(pid)],capture_output=True,text=True).stdout.strip())
