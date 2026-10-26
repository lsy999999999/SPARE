"""Resumable, detached CPU evaluation controller with atomic progress."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import fcntl
import gzip
import json
import os
import statistics
import sys
import time
import traceback
from .common import ROOT, HERE, OUT, read, write, digest


def run_task(task):
    from .simulate import simulate
    from .timing import FittedTiming
    name=task['name'];dest=OUT/'runs'/name
    dest.mkdir(parents=True,exist_ok=True)
    write(dest/'running.json',dict(pid=os.getpid(),started=time.time(),task=name))
    try:
        options=dict(task['options']);engine_arm=options.pop('engine_arm','arrival')
        profile=task['profile'];timing=FittedTiming(read(OUT/'timing_profile.json')) if profile=='gpu_fitted' else None
        history=[]
        if task.get('history'):
            history=[json.loads(line)['actual_output_tokens'] for line in (ROOT/task['history']).read_text().splitlines() if line.strip()]
        result=simulate(read(ROOT/task['input']),arm=engine_arm,capacity=task['capacity'],
                        profile='resource' if timing else profile,policy_options=options,
                        fitted_timing=timing,history=history,output=dest)
        # Compact iteration records retain the quantities needed for queue and guard audits.
        thresholds=[];counts=Counter();pending_peak=unfinished_peak=0;integral=0.;stop_backlog=None
        reqs=list(csv.DictReader((dest/'requests.csv').open()))
        last_arrival=max(float(r['arrival_ms']) for r in reqs)
        with gzip.open(dest/'iterations.jsonl.gz','rt') as h:
            for line in h:
                row=json.loads(line)
                if row.get('threshold') is not None:thresholds.append(row['threshold'])
                for key in ('rejected_guard','rejected_saving','rejected_infeasible','candidates'):
                    counts[key]+=row.get(key,0)
                for r in row.get('reductions',[]):
                    if r['outputs']==0 and r['baseline_tokens']==r['remaining']:
                        counts['finish_opportunities_deferred']+=1
                    if r['outputs']>0 and r['remaining']==1:counts['normal_decode_reductions']+=1
                base=row.get('baseline_growth_blocks')
                if base is not None:counts['saved_block_events']+=max(0,base-row['actual_growth_blocks'])
                pending_peak=max(pending_peak,row['pending_first_tokens'])
                unfinished_peak=max(unfinished_peak,row['unfinished_requests'])
                if stop_backlog is None and row['simulated_start_ms']>=last_arrival:
                    stop_backlog=row['pending_first_tokens']
        metrics=result['metrics']
        metrics.update(counts)
        metrics.update(pending_first_token_peak=pending_peak,unfinished_peak=unfinished_peak,
                       pending_first_tokens_after_last_arrival=stop_backlog or 0,
                       drain_ms=max(float(r['finish_ms']) for r in reqs)-last_arrival,
                       threshold_mean=statistics.fmean(thresholds) if thresholds else None)
        for q in range(4):
            group=reqs[q*len(reqs)//4:(q+1)*len(reqs)//4]
            metrics[f'arrival_quartile{q}_ttft_ms']=statistics.fmean(float(r['ttft_ms']) for r in group)
        result.update(task=task,arm=task['arm'],profile=profile,input_sha256=digest(ROOT/task['input']),
                      timing_claim='retrospective GPU-fitted simulation; see validation' if timing else result['timing_claim'],
                      timing_profile_sha256=digest(OUT/'timing_profile.json') if timing else None,
                      full_source_sha256=read(OUT/'design.json')['source_hashes'])
        write(dest/'result.json',result)
        (dest/'running.json').unlink(missing_ok=True)
        return dict(name=name,family=task['family'],complete=True,wall_s=metrics['cpu_wall_s'])
    except Exception:
        error=dict(name=name,family=task['family'],complete=False,error=traceback.format_exc(),time=time.time())
        write(dest/'error.json',error)
        (dest/'running.json').unlink(missing_ok=True)
        return error


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=12);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();pid=os.getpid()
    def status(stage,**kw):
        write(OUT/'status.json',dict(stage=stage,pid=pid,started=start,updated=time.time(),jobs=args.jobs,**kw))
    try:
        status('preparing')
        from .prepare import prepare
        design=prepare()
        for rel,sha in design['source_hashes'].items():
            assert digest(ROOT/rel)==sha,('Frozen source changed; use a new batch',rel)
        status('calibrating',expected=design['expected'])
        from .timing import calibrate
        calibrate()
        done=Counter();failed=[];pending=[]
        for task in design['tasks']:
            p=OUT/'runs'/task['name']/'result.json'
            if p.exists():
                r=read(p)
                if r.get('complete') and r.get('task')==task and r.get('full_source_sha256')==design['source_hashes']:
                    done[task['family']]+=1;continue
            pending.append(task)
        finished=sum(done.values())
        print('Matrix starts:',design['by_family'],'workers',args.jobs,flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures={pool.submit(run_task,t):t for t in pending}
            while futures:
                ready,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
                for future in ready:
                    task=futures.pop(future)
                    try:r=future.result()
                    except Exception:r=dict(name=task['name'],family=task['family'],complete=False,error=traceback.format_exc())
                    if r['complete']:done[r['family']]+=1;finished+=1
                    else:failed.append(r);print('FAILED',r,flush=True)
                    if finished%25==0:print(f'{finished}/{design["expected"]} complete; failures {len(failed)}; elapsed {(time.time()-start)/60:.1f} min',flush=True)
                status('running',completed=finished,expected=design['expected'],failed=len(failed),
                       by_family=dict(done),expected_by_family=design['by_family'],remaining=len(futures))
        write(OUT/'failures.json',failed)
        status('analyzing',completed=finished,expected=design['expected'],failed=len(failed))
        from .analyze import analyze
        analyze()
        from .overhead import benchmark
        benchmark()
        # Update report with the independent CPU microbenchmark results.
        analyze()
        stage='complete' if not failed else 'complete_with_errors'
        status(stage,completed=finished,expected=design['expected'],failed=len(failed),by_family=dict(done),
               elapsed_s=time.time()-start,report=str(OUT/'REPORT_CN.md'))
        print(stage,finished,'/',design['expected'],'report',OUT/'REPORT_CN.md',flush=True)
    except Exception:
        status('controller_failed',error=traceback.format_exc())
        raise


if __name__=='__main__':main()
