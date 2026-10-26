import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/v1_efficient_points_20260927_r2'


def read(p):return json.loads(p.read_text())


def main():
    if not (OUT/'status.json').exists():
        print('Controller not started; see launcher.log (CPU validation / startup).');return
    status=read(OUT/'status.json')
    print('State:',status['state'],'phase:',status.get('phase'))
    formal=0;probes=0
    for folder in sorted((OUT/'groups').glob('*')):
        s=read(folder/'status.json') if (folder/'status.json').exists() else {}
        cur=read(folder/'current.json') if (folder/'current.json').exists() else {}
        if folder.name.startswith('profile_'):
            count=len(list((folder/'profile').glob('*.json')));probes+=count
            print(folder.name,'gpu',s.get('gpu'),s.get('state'),f'{count}/720 probes',cur.get('context'),cur.get('q'))
        else:
            runs=[read(p) for p in (folder/'runs').glob('*.json')]
            count=sum(r.get('phase')=='formal' and r.get('audit',{}).get('passed',False) for r in runs);formal+=count
            print(folder.name,'gpu',s.get('gpu'),s.get('state'),f'{count}/20 formal',cur.get('name'))
        if s.get('error'):print(s['error'][-1500:])
    print(f'Saved timing probes: {probes}/2880 (includes warmups); audited formal runs: {formal}/120')
    print('Pending:',status.get('pending'));print('Finished:',status.get('finished'))
    if (OUT/'calibration.json').exists():print('Frozen pairs:',read(OUT/'calibration.json')['pairs'])


if __name__=='__main__':main()
