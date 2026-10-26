"""Compact read-only progress; safe to run repeatedly via watch."""
from datetime import datetime
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_saving_threshold_20260924.common import OUT,read

print(datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))
p=OUT/'status.json'
s=read(p) if p.exists() else {}
pid=s.get('pid')
try:alive=bool(pid and b'v1_saving_threshold_20260924/run.py' in Path(f'/proc/{pid}/cmdline').read_bytes())
except (FileNotFoundError,ProcessLookupError):alive=False
print('Controller:',s.get('state','not_started'),'PID:',pid,'alive:',alive)
p=OUT/'pipeline_status.json'
if p.exists():
    pipeline=read(p)
    print('Pipeline:',pipeline.get('state'),'PID:',pipeline.get('pid'))
p=OUT/'finalize_status.json'
if p.exists():
    finalizer=read(p)
    print('Finalizer:',finalizer.get('state'),'PID:',finalizer.get('pid'))
p=OUT/'inputs_manifest.json'
jobs=list(read(p)['jobs']) if p.exists() else []
for p in sorted((OUT/'plans').glob('*.json')):jobs.extend(read(p)['jobs'])
for phase in sorted({j['phase'] for j in jobs}):
    selected=[j for j in jobs if j['phase']==phase]
    rows=[read(p) for j in selected if (p:=OUT/'runs'/(j['key']+'.json')).exists()]
    complete=sum(r['status']=='complete' for r in rows)
    print(f'{phase}: complete={complete}/{len(selected)}, failed={len(rows)-complete}, pending={len(selected)-len(rows)}')
for p in sorted(OUT.glob('worker_g*.json')):
    w=read(p);j=w.get('job',{})
    print(p.stem,w.get('state'),j.get('key',w.get('task','')))
p=OUT/'cpu/status.json'
cpu=read(p) if p.exists() else {}
print('CPU:',cpu.get('state','not_started'),'records:',len(list((OUT/'cpu/runs').glob('*.json'))))
print('Errors:',len(list((OUT/'errors').glob('*.json'))),'Contamination:',len(list((OUT/'contamination').glob('*.json'))))
print('Output:',OUT)
