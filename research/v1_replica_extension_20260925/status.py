"""Read current extension queue, with no stale original deadline."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_replica_extension_20260925.common import *

print(time.strftime('%Y-%m-%d %H:%M:%S %z'))
for name in ('supervisor_status.json','status.json','analysis/summary.json'):
    path=OUT/name
    if not path.exists():continue
    value=read(path)
    if name=='supervisor_status.json':
        value=dict(value,pending_count=len(value.get('pending',[])))
        value['pending']=value.get('pending',[])[:3]
    if name=='analysis/summary.json':value={k:v for k,v in value.items() if k!='mechanics'}
    print(name,value)
for root in (ROOT/'artifacts/v1_mechanism_models_replicas_20260925_r1',OUT):
    print(root.name)
    for f in sorted((root/'groups').glob('*/status.json')):
        s=read(f)
        if s['state']=='running':
            excluded=any((f.parent/name).exists() for name in ('contamination.json','interrupted.json','controller_failure.json'))
            print(f.parent.name,'excluded_interrupted_group' if excluded else 'group_status_running',
                  s,'saved_global_episodes',len(list((f.parent/'runs').glob('*.json'))))
supplement=ROOT/'artifacts/v1_replica_extension_20260926_supplement_r1/status.json'
if supplement.exists():print('supplement',read(supplement))
