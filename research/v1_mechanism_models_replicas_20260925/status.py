"""Read-only status for the pilot and automatic mechanism pipeline."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_mechanism_models_replicas_20260925.common import *


def main():
    print(time.strftime('%Y-%m-%d %H:%M:%S %z'))
    for relative in ('status.json','analysis/status.json'):
        path=OUT/relative
        print(relative,read(path) if path.exists() else 'not yet available')
    if (OUT/'launch.json').exists():
        launch=read(OUT/'launch.json')
        print('Elapsed minutes:',round((time.time()-launch['time'])/60,1))
        if (OUT/'deadline_amendment.json').exists():
            print('Total deadline removed by user amendment:',read(OUT/'deadline_amendment.json'))
        else:
            print('Remaining deadline minutes:',round(max(0,launch['deadline']-time.time())/60,1))
    formal=0
    for path in (OUT/'groups').glob('*/runs/*.json'):
        result=read(path)
        formal+=result['job']['phase']=='compare'
    print('All saved formal global episodes (includes partial/failed groups):',formal)
    if (OUT/'analysis/summary.json').exists():
        summary=read(OUT/'analysis/summary.json')
        print('Accepted complete groups:',len(summary['accepted_groups']),
              'Accepted global episodes:',summary['formal_global_episodes'])


if __name__=='__main__':main()
