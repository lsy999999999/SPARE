"""Frozen matrix mechanism replay, never labeled GPU performance."""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.www_baseline_fidelity_20261013.common import *
sys.path.insert(0,str(OUT/'source'))


def task(cell,arm,profile):
    from research.www_baseline_fidelity_20261013.simulate import run
    folder=OUT/'simulation'/profile/cell['key']/arm
    result=run(read(OUT/cell['input']),arm,cell['capacity'],read(OUT/cell['history_file'])['lengths'],profile=profile,output=folder)
    return dict(cell=cell['key'],arm=arm,profile=profile,metrics=result['metrics'])


def main():
    design=read(OUT/'design.json')
    done=[];failed=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks={pool.submit(task,c,a,p):(c['key'],a,p) for c in design['cells'] for a in ARMS for p in ('resource','resource_staircase')}
        for future in as_completed(tasks):
            try:done.append(future.result())
            except Exception:failed.append(dict(task=tasks[future],error=traceback.format_exc()))
            write(OUT/'simulation/status.json',dict(state='running',completed=len(done),total=len(tasks),failed=failed,updated=time.time()))
    write(OUT/'simulation/results.json',done)
    write(OUT/'simulation/status.json',dict(state='complete' if not failed else 'failed',completed=len(done),total=len(tasks),failed=failed,updated=time.time()))
    print('simulation',len(done),'failed',len(failed))

if __name__=='__main__':main()
