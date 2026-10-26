import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_candidate_audit_20260928.common import *

for name in ('launcher.json','status.json'):
    if (OUT/name).exists():print(name,json.dumps(read(OUT/name),ensure_ascii=False))
for folder in sorted((OUT/'groups').glob('*')):
    status=read(folder/'status.json') if (folder/'status.json').exists() else {}
    current=read(folder/'current.json') if (folder/'current.json').exists() else {}
    print(folder.name,status.get('state'),len(list((folder/'runs').glob('*.json'))),current.get('name'))
    if status.get('state')=='failed':print(status['error'])
