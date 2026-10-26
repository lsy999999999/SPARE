"""Read-only progress summary, runnable on either host."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[3]
out=root/'artifacts/v1_ratio_closed_loop_20260928_r1'
def read(p):return json.loads(p.read_text())
status=read(out/'status.json') if (out/'status.json').exists() else {}
print('controller',status.get('state'),'finished',len(status.get('finished',[])),'pending',len(status.get('pending',[])))
formal=aa=0
for folder in sorted((out/'groups').glob('*')):
    saved=list((folder/'runs').glob('*.json'));formal+=sum(p.stem.startswith(('steady','burst')) for p in saved);aa+=sum(p.stem.startswith('aa') for p in saved)
    s=read(folder/'status.json') if (folder/'status.json').exists() else {}
    c=read(folder/'current.json') if (folder/'current.json').exists() else {}
    ex=read(folder/'exclusion.json') if (folder/'exclusion.json').exists() else {}
    print(folder.name,s.get('state'),c.get('name'),c.get('state'),'saved',len(saved),ex)
    if s.get('error'):print(s['error'])
print('saved formal',formal,'/192, aa',aa,'/32')
if (out/'simulation_status.json').exists():print('simulation',read(out/'simulation_status.json'))
