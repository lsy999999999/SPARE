"""Usage: python3 -m research.www_evaluation_completion_20261003.status"""
from datetime import datetime
import os
import time
from .common import OUT, read

def main():
    if not (OUT/'status.json').exists():print('Not started:',OUT);return
    s=read(OUT/'status.json')
    try:os.kill(s['pid'],0);alive=True
    except (OSError,KeyError):alive=False
    print('Stage:',s['stage'],'| controller PID:',s.get('pid'),'| alive:',alive)
    print('Complete:',s.get('completed',0),'/',s.get('expected','?'),'| failures:',s.get('failed',0),
          '| CPU workers:',s.get('jobs'))
    print('Elapsed: %.1f min | last update: %.1f s ago'%((time.time()-s['started'])/60,time.time()-s['updated']))
    for name,n in s.get('expected_by_family',{}).items():print(' ',name,s.get('by_family',{}).get(name,0),'/',n)
    print('Log:',OUT/'controller.log')
    if s.get('report'):print('Report:',s['report'])
    if s.get('error'):print(s['error'])

if __name__=='__main__':main()
