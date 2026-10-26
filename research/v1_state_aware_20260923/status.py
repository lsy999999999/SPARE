"""Progress query uses stdlib only; safe while the experiment is running."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_state_aware_20260923.common import OUT, read


def main():
    status=read(OUT/'status.json') if (OUT/'status.json').exists() else {'state':'not_started'}
    pid=status.get('pid'); alive=False
    if pid:
        try: os.kill(pid,0); alive=True
        except ProcessLookupError: pass
    counts={}; failed=[]
    for p in (OUT/'runs').glob('*.json'):
        r=read(p); key=r['job']['phase']+'/'+r['status']
        counts[key]=counts.get(key,0)+1
        if r['status']!='complete': failed.append(r['job']['key'])
    m=read(OUT/'manifest.json') if (OUT/'manifest.json').exists() else {}
    value=dict(controller=status,controller_alive=alive,counts=counts,failed=failed,
        expected_compare=72,expected_validation=m.get('validation_expected'),
        simulation=read(OUT/'simulation_status.json') if (OUT/'simulation_status.json').exists() else None,
        workers={p.stem:read(p) for p in OUT.glob('worker_g*.json')},now=time.time())
    print(json.dumps(value,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
