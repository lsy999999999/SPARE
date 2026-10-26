"""Cheap read-only dashboard; no model, CUDA or multi-GB round logs."""
from pathlib import Path
import json,time,subprocess
from datetime import datetime
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'artifacts/v1_broad12h_20260925_r1'
def read(p):return json.loads(p.read_text())
if __name__=='__main__':
    now=time.time();print(datetime.now().astimezone().isoformat(timespec='seconds'))
    for name in ('status.json','analysis/status.json','simulator/status.json'):
        if (OUT/name).exists():print(name,read(OUT/name))
    if (OUT/'launch.json').exists():
        launch=read(OUT/'launch.json');elapsed=(now-launch['time'])/3600
        print('Elapsed hours:',round(elapsed,2),'Deadline:',datetime.fromtimestamp(launch['deadline']).astimezone().isoformat(timespec='seconds'),'remaining hours:',round(max(0,launch['deadline']-now)/3600,2))
        if launch.get('pid'):print(subprocess.run(['ps','-o','pid,ppid,sid,etime,args','-p',str(launch['pid'])],capture_output=True,text=True).stdout.strip())
    if (OUT/'design.json').exists():
        d=read(OUT/'design.json');entries=[read(p) for p in (OUT/'groups').glob('*.json')];valid=[r for r in entries if r['state']=='complete']
        print('Complete valid groups:',len(valid),'/',len(d['cells']),'Formal runs in accepted groups:',len(valid)*len(d['methods'])*2,'/',d['expected_formal_runs'])
        count=sum(1 for p in (OUT/'attempts').glob('*/a*/runs/*.json'))
        print('All saved formal attempt records (includes active/rejected):',count)
        if valid and (OUT/'launch.json').exists() and elapsed>0:
            hours=(len(d['cells'])-len(valid))/(len(valid)/elapsed)
            print('Crude remaining estimate from completed groups:',round(hours,1),'hours; changes with workload and idle cards')
    for p in sorted(OUT.glob('worker_g*.json')):print(p.stem,read(p))
    cals=[read(p) for p in (OUT/'calibration').glob('*/status.json')]
    print('GPU/capacity calibrations:',len(cals),'passed:',sum(c['passed'] for c in cals))
    if (OUT/'analysis/summary.json').exists():
        s=read(OUT/'analysis/summary.json');print('Audit:',s['audit'],'matrix complete:',s['complete_matrix'])
