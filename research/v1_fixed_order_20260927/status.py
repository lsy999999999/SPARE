import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/v1_fixed_order_20260927_r1'


def main():
    if not (OUT/'status.json').exists():
        print('Controller status not yet available');return
    status=json.loads((OUT/'status.json').read_text())
    print('state:',status['state'])
    print('accepted groups:',sum(r.get('accepted',False) for r in status.get('finished',[])),'/6')
    count=0
    for folder in sorted((OUT/'groups').glob('*')):
        jobs=[json.loads(p.read_text()) for p in (folder/'runs').glob('*.json')]
        formal=sum(r.get('phase')=='formal' and r.get('status')=='complete' for r in jobs)
        branch=sum(r.get('phase')=='branch' and r.get('prefix_matched',False) for r in jobs)
        count+=formal
        cur=json.loads((folder/'current.json').read_text()) if (folder/'current.json').exists() else {}
        print(folder.name,'formal',formal,'/24','branches',branch,'last:',cur.get('name'),cur.get('state'))
    print('saved formal runs:',count,'/144 (acceptance requires complete clean group)')
    print('running:',status.get('running',{}));print('pending:',status.get('pending',[]))


if __name__=='__main__':main()
