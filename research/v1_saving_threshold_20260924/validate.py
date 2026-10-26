"""Native CPU differential checks and resumable synchronous allocator replays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import os
import subprocess
import fcntl
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_saving_threshold_20260924.common import *
from research.v1_saving_threshold_20260924.scheduler import FactorScheduler
from research.v1_state_aware_20260923.scheduler import StateScheduler
from research.v1_budget_20260922.test_scheduler import scheduler, add, advance
from research.v1_budget_20260922.policy import State
from research.v1_saving_threshold_20260924.policy import select


def equivalence():
    count = 0
    for cap in (48,96,160):
        for n,old_method in ((1,'a_any_positive_saving'),('off','srp_matched')):
            a = scheduler(blocks=cap,cls=FactorScheduler)
            b = scheduler(blocks=cap,cls=StateScheduler)
            a.configure('threshold',256,256,8,threshold=n)
            b.configure(old_method,256,256,8)
            for s in (a,b): add(s,[257,129,385,99,400,12])
            for _ in range(10000):
                x,y=a.schedule(),b.schedule()
                assert x.num_scheduled_tokens==y.num_scheduled_tokens
                for field in ('free_after','preemptions','recomputed_tokens'):
                    assert a.round_rows[-1][field]==b.round_rows[-1][field]
                advance(a,x);advance(b,y)
                if not a.requests: break
            assert not a.requests and not b.requests
            count += 1
    rows=[State(str(i),0,1024,1024,0,0,False) for i in range(8)]
    for n in (1,4,16,32,64,1000,'off'):
        a=select(rows,1024,256,1000,1024,16,32,12288,n)
        if a['changed']: assert a['base'].growth-a['selected'].growth>=n
    result=dict(passed=count,threshold_boundary_checks=7,time=time.time())
    write(OUT/'cpu/equivalence.json',result)
    return result


def replay(case,cap,n):
    from vllm import SamplingParams
    from vllm.v1.request import Request,RequestStatus
    s=scheduler(blocks=cap,cls=FactorScheduler)
    s.configure('threshold',**BASE,threshold=n)
    inputs=read(INPUT/'traces'/f'{case}.json')
    targets={r['id']:r['output'] for r in inputs}
    for i,r in enumerate(inputs):
        s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,
            SamplingParams(max_tokens=r['output']),None,None,arrival_time=float(i)))
    first,finished={},{}
    h=hashlib.sha256();start=time.monotonic()
    for step in range(1,100001):
        out=s.schedule(); h.update(json.dumps(out.num_scheduled_tokens,sort_keys=True).encode())
        done=[]
        for rid in out.num_scheduled_tokens:
            r=s.requests[rid]
            if r.num_computed_tokens==r.num_tokens:
                r.append_output_token_ids(100);first.setdefault(rid,step)
            if len(r.output_token_ids)==targets[rid]:done.append(rid);finished[rid]=step
        s.finish_requests(done,RequestStatus.FINISHED_LENGTH_CAPPED)
        if not s.requests or time.monotonic()-start>180:break
    return dict(case=case,capacity=cap,threshold=n,complete=not s.requests,
        requests=len(inputs),finished=len(finished),rounds=len(s.round_rows),
        first_round=first,finish_round=finished,action_sha256=h.hexdigest(),
        preemptions=sum(r['preemptions'] for r in s.round_rows),
        recomputed_tokens=sum(r['recomputed_tokens'] for r in s.round_rows),
        interventions=sum(r['intervened'] for r in s.round_rows),elapsed_s=time.monotonic()-start,
        interpretation='synchronous native allocation; rounds are not GPU milliseconds')


def main(full):
    lockpath=OUT/'cpu/controller.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    lock=lockpath.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(OUT/'cpu/status.json',dict(state='running',pid=os.getpid(),time=time.time()))
    print(equivalence(),flush=True)
    if full:
        for cap in (512,768,2048,4096,256,384):
            for case in ('code_rate1','conversation_rate1'):
                rows=read(INPUT/'traces'/f'{case}.json')
                for n in THRESHOLDS:
                    p=OUT/'cpu/runs'/f'{case}_c{cap}_n{n}.json'
                    if p.exists():continue
                    if max(r['prompt']+r['output'] for r in rows)>(cap-1)*16:
                        row=dict(case=case,capacity=cap,threshold=n,status='infeasible_single_request')
                    else:row=replay(case,cap,n)
                    write(p,row); print(p.name,row.get('complete',row.get('status')),flush=True)
        write(OUT/'cpu/status.json',dict(state='finished',time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--launch',action='store_true')
    a=p.parse_args()
    if a.launch:
        (OUT/'cpu').mkdir(parents=True,exist_ok=True)
        with (OUT/'cpu/controller.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--full'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        print('Detached CPU controller',child.pid)
    else:main(a.full)
