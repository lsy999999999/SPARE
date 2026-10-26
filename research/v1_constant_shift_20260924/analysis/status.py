"""Read-only status; works with system Python, no CUDA imports."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/v1_constant_shift_20260924_r1'


def read(path):return json.loads(path.read_text())


def main():
    print(datetime.now().astimezone().isoformat(timespec='seconds'))
    status=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
    print('Controller:',status.get('state','not started'),'Phase:',status.get('phase','-'))
    runs=[read(p) for p in (OUT/'runs').glob('*.json')]
    counts=Counter((r['job']['phase'],r['status']) for r in runs)
    print('Completed:',sum(r['status']=='complete' for r in runs),'Failed:',sum(r['status']!='complete' for r in runs))
    print('By phase:',dict(counts))
    m=read(OUT/'manifest.json') if (OUT/'manifest.json').exists() else {}
    gs=m.get('gpus',[])
    print('Plan:',56,'paired comparison runs +',2*len(gs),'A/A =',56+2*len(gs),'total; GPUs:',gs)
    for p in sorted(OUT.glob('worker_g*.json')):
        w=read(p);print(p.stem,w.get('state'),w.get('job',''))
    for p in sorted(OUT.glob('calibration_g*.json')):
        a=read(p);print(p.stem,'PASS' if a['passed'] else 'FAIL','TTFT/elapsed spread:',','.join(f'{v:.2%}' for v in a['ratios']))
    if (OUT/'design.json').exists():
        d=read(OUT/'design.json');print('Frozen matched constant:',d['constant_delta'],'; other constant: 32')
    if (OUT/'frozen_formula.json').exists():
        f=read(OUT/'frozen_formula.json');print('Frozen:',f['winner'],f['spec']);print('Development losses:',f['scores'])
    if (OUT/'analysis/status.json').exists():print('Report:',read(OUT/'analysis/status.json'))
    if status.get('error'):print('ERROR:',status['error'])
    if runs:
        print('Latest results:')
        for r in sorted(runs,key=lambda r:r['saved_at'])[-6:]:
            print(' ',r['job']['key'],r['status'],'TTFT(ms)=',r['metrics']['mean_ttft_ms'])
    print('Live controller (empty means finished/stopped):')
    proc=subprocess.run(['pgrep','-af',str(ROOT/'research/v1_constant_shift_20260924/run.py')+'$'],capture_output=True,text=True)
    print(proc.stdout.strip() or '(none)')


if __name__=='__main__':main()
