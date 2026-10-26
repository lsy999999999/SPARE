"""Actual shared-ingress GPU experiment, detached and bounded to idle GPUs."""
import argparse
import gzip
import importlib.metadata
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import shutil
import signal
import statistics as st
import subprocess
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.v1_replica_extension_20260925.common import *
from research.v1_state_aware_20260923 import run as infra
from research.v1_budget_20260922.run import quantile


def summarize(clocks, rounds, elapsed):
    ttft = [c['first_ms'] - c['arrival_ms'] for c in clocks.values()]
    e2e = [c['finish_ms'] - c['arrival_ms'] for c in clocks.values()]
    tbt = [b-a for c in clocks.values() for a,b in zip(c['token_times_ms'], c['token_times_ms'][1:])]
    result = dict(mean_ttft_ms=st.fmean(ttft), p95_ttft_ms=quantile(ttft, .95),
                  mean_e2e_ms=st.fmean(e2e), p95_e2e_ms=quantile(e2e, .95),
                  mean_tbt_ms=st.fmean(tbt) if tbt else None, p95_tbt_ms=quantile(tbt,.95),
                  throughput_tokens_s=sum(c['tokens'] for c in clocks.values())/(elapsed/1000),
                  completion_rate=1.0, waiting_area_ms=sum(ttft), round_count=len(rounds))
    for metric, field in [('preemptions','preemptions'), ('recomputed_tokens','recomputed_tokens'),
                          ('interventions','intervened')]:
        result[metric] = sum(r[field] for r in rounds)
    for field in ('decision_ms','audit_ms','schedule_ms','step_ms'):
        xs = [r[field] for r in rounds if field in r]
        result[field+'_total'] = sum(xs)
        result[field+'_mean'] = st.fmean(xs) if xs else None
    result['submission_lag_ms'] = st.fmean(c['submitted_ms']-c['arrival_ms'] for c in clocks.values())
    return result


def audit(reqs, clocks, rounds):
    assert set(clocks) == {r['id'] for r in reqs}
    for r in reqs:
        c = clocks[r['id']]
        assert c['tokens'] == r['output'] == len(c['token_times_ms'])
        assert c['arrival_ms'] == r['arrival_ms']
        assert 0 <= c['arrival_ms'] <= c['dispatched_ms'] <= c['submitted_ms'] <= c['first_ms'] <= c['finish_ms']
        assert c['first_ms'] == c['token_times_ms'][0]
        assert all(a <= b for a,b in zip(c['token_times_ms'],c['token_times_ms'][1:]))
    from research.v1_budget_20260922.policy import State, project
    for index, row in enumerate(rounds):
        assert row['round'] == index
        assert row['free_before'] + sum(r['blocks'] for r in row['state']) == row['total_blocks']-1
        assert 0 <= row['free_after'] <= row['total_blocks']-1
        assert sum(row['scheduled'].values()) <= row['budget']
        assert row['step_ms']+1e-6 >= row['schedule_ms']
        if row.get('verified'):
            plan = project([State(**r) for r in row['state']], row['budget'], row['chunk'],
                           row['free_before'], 16, 32, 8176)
            # Paper admission can temporarily hide waiting rows; its own exact
            # execution audit remains active. Re-project formula paths here.
            if row['method'] == 'threshold':
                assert plan is not None and plan.tokens == row['scheduled']
                assert plan.growth == row['free_before']-row['free_after']


def worker(rank, gpu, task, control, incoming):
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                          VLLM_USE_V1='1', VLLM_ENABLE_V1_MULTIPROCESSING='0',
                          OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        manifest = read(OUT/'manifest.json')
        os.sched_setaffinity(0, manifest['affinity'][str(gpu)])
        design = read(OUT/'design.json')
        import torch
        from vllm import SamplingParams
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        actual_uuid = str(torch.cuda.get_device_properties(0).uuid).lower()
        assert actual_uuid == manifest['hardware'][str(gpu)]['uuid'].removeprefix('GPU-').lower()
        engine = LLMEngine.from_engine_args(EngineArgs(
            model=design['models'][task['model']]['path'], dtype='half', max_model_len=8176,
            max_num_seqs=64, max_num_batched_tokens=2048, enable_chunked_prefill=True,
            enable_prefix_caching=False, block_size=16, gpu_memory_utilization=.9,
            enforce_eager=True, disable_log_stats=True, seed=0,
            num_gpu_blocks_override=task['capacity'],
            scheduler_cls='research.v1_arrival_kv_papers_20260925.scheduler.MatrixScheduler'))
        scheduler = engine.engine_core.engine_core.scheduler
        from research.v1_budget_20260922.run import episode
        warm = [dict(id=f'w{i}', prompt=256, output=16, arrival_ms=0) for i in range(4)]
        for _ in range(2):
            assert episode(engine,scheduler,warm,'srp',BASE,60)['status']=='complete'
        control.send(dict(state='loaded',rank=rank,pid=os.getpid(),gpu=gpu))
        while True:
            message = control.recv()
            if message['op'] == 'stop': break
            job = message['job']; reqs = message['requests']
            scheduler.history_context = read(OUT/'histories.json')[job['source']]['lengths']
            scheduler.seed_context = 2026092500 + job['repeat'] + 1000*rank
            spec = ARMS[job['arm']]
            scheduler.configure(spec['method'], **spec['config'])
            control.send(dict(state='ready', key=job['key'], rank=rank))
            start = control.recv()['start']
            ended = False; clocks = {}; error = None
            while not ended or engine.has_unfinished_requests():
                now = (time.perf_counter()-start)*1000
                if now > design['episode_timeout_s']*1000:
                    raise TimeoutError('Predeclared episode deadline')
                while True:
                    try: packet = incoming.get_nowait()
                    except queue.Empty: break
                    assert packet['key'] == job['key']
                    if packet['op']=='end': ended=True; break
                    r = packet['request']
                    assert r['id'] not in clocks
                    clocks[r['id']] = dict(arrival_ms=r['arrival_ms'],dispatched_ms=packet['dispatched_ms'],
                        submitted_ms=(time.perf_counter()-start)*1000, first_ms=None, finish_ms=None,
                        tokens=0, token_times_ms=[], replica=rank)
                    engine.add_request(r['id'],{'prompt_token_ids':[100]*r['prompt']},
                        SamplingParams(temperature=0.,max_tokens=r['output'],ignore_eos=True,detokenize=False))
                if engine.has_unfinished_requests():
                    step_start = time.perf_counter()
                    outputs = engine.step()
                    now = (time.perf_counter()-start)*1000
                    row = scheduler.round_rows[-1]
                    row.update(step_ms=(time.perf_counter()-step_start)*1000,
                               step_start_ms=(step_start-start)*1000,step_end_ms=now)
                    for output in outputs:
                        c=clocks[output.request_id]
                        n=len(output.outputs[0].token_ids) if output.outputs else 0
                        if n>c['tokens']:
                            assert n==c['tokens']+1
                            c['tokens']=n;c['token_times_ms'].append(now)
                            if c['first_ms'] is None:c['first_ms']=now
                        if output.finished:c['finish_ms']=now
                elif not ended: time.sleep(.0005)
            rounds = scheduler.round_rows
            audit(reqs, clocks, rounds)
            elapsed = max(c['finish_ms'] for c in clocks.values())
            folder = OUT/'groups'/task['key']
            path = folder/'rounds'/f"{job['key']}_replica{rank}.json.gz"
            path.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(path,'wt',compresslevel=1) as stream:
                json.dump(rounds,stream,separators=(',',':'))
            result=dict(status='complete',job=job,task=task,rank=rank,gpu=gpu,requests=reqs,clocks=clocks,
                        elapsed_ms=elapsed,metrics=summarize(clocks,rounds,elapsed),
                        rounds_path=str(path.relative_to(OUT)),rounds_sha256=digest(path))
            target=folder/'replicas'/f"{job['key']}_replica{rank}.json"
            write(target,result)
            control.send(dict(state='complete',path=str(target.relative_to(OUT)),rank=rank))
        control.close()
    except BaseException:
        error=traceback.format_exc()
        try: control.send(dict(state='failed',rank=rank,error=error))
        except Exception: pass
        raise


def receive_all(connections, processes, expected, timeout):
    deadline=time.monotonic()+timeout; pending=set(range(len(connections))); results={}
    while pending:
        for i in list(pending):
            if connections[i].poll():
                result=connections[i].recv()
                assert result['state']==expected,result
                results[i]=result;pending.remove(i)
            elif not processes[i].is_alive(): raise RuntimeError(f'Worker {i} exited unexpectedly')
        if time.monotonic()>deadline:raise TimeoutError(f'Waiting for {expected}')
        if pending:time.sleep(.005)
    return [results[i] for i in range(len(connections))]


def group(taskkey, gpus):
    manifest=read(OUT/'manifest.json');design=read(OUT/'design.json')
    assert sources()==manifest['sources']
    assert digest(OUT/'design.json')==manifest['design_sha256']
    assert digest(OUT/'histories.json')==design['histories_sha256']
    for case,sha in design['trace_hashes'].items():assert digest(OUT/'inputs'/f'{case}.json')==sha
    task=next(t for t in design['tasks'] if t['key']==taskkey)
    assert len(gpus)==task['replicas']
    locks=[infra.acquire(f'/tmp/query-capacity-gpu{g}.lock') for g in sorted(gpus)]
    assert all(infra.idle(infra.hardware()[g]) for g in gpus),'GPU no longer idle'
    os.sched_setaffinity(0,manifest['ingress_affinity'][str(gpus[0])])
    folder=OUT/'groups'/taskkey
    assert not (folder/'status.json').exists(),'No overwrite/restart of measured groups'
    ctx=mp.get_context('spawn');connections=[];queues=[];processes=[]
    try:
        for rank,gpu in enumerate(gpus):
            parent,child=ctx.Pipe();incoming=ctx.Queue()
            process=ctx.Process(target=worker,args=(rank,gpu,task,child,incoming))
            process.start();child.close()
            connections.append(parent);queues.append(incoming);processes.append(process)
        receive_all(connections,processes,'loaded',600)
        write(folder/'status.json',dict(state='running',gpus=gpus,worker_pids=[p.pid for p in processes],updated=time.time()))

        def execute(job,reqs):
            mapping=routes(reqs,len(gpus))
            for rank,conn in enumerate(connections):
                conn.send(dict(op='run',job=job,requests=[r for r in reqs if mapping[r['id']]==rank]))
            receive_all(connections,processes,'ready',30)
            start=time.perf_counter()+.05
            for conn in connections:conn.send(dict(start=start))
            cursor=0;dispatch=[]
            while cursor<len(reqs):
                now=(time.perf_counter()-start)*1000
                if now>=reqs[cursor]['arrival_ms']:
                    r=reqs[cursor];rank=mapping[r['id']]
                    queues[rank].put(dict(op='request',key=job['key'],request=r,dispatched_ms=now))
                    dispatch.append(dict(id=r['id'],replica=rank,arrival_ms=r['arrival_ms'],dispatched_ms=now))
                    cursor+=1
                else:time.sleep(min(.001,(reqs[cursor]['arrival_ms']-now)/1000))
            for q in queues:q.put(dict(op='end',key=job['key']))
            outputs=receive_all(connections,processes,'complete',design['episode_timeout_s']+30)
            records=[read(OUT/r['path']) for r in outputs]
            clocks={rid:c for r in records for rid,c in r['clocks'].items()}
            assert len(clocks)==len(reqs) and all(c['replica']==mapping[rid] for rid,c in clocks.items())
            elapsed=max(c['finish_ms'] for c in clocks.values())
            metrics=summarize(clocks,[],elapsed)
            for field in ('preemptions','recomputed_tokens','interventions','round_count',
                          'decision_ms_total','audit_ms_total','schedule_ms_total','step_ms_total'):
                metrics[field]=sum(r['metrics'][field] for r in records)
            for field in ('decision_ms','audit_ms','schedule_ms','step_ms'):
                metrics[field+'_mean']=metrics[field+'_total']/max(1,metrics['round_count'])
            result=dict(status='complete',task=task,job=job,gpus=gpus,clocks=clocks,requests=reqs,
                        metrics=metrics,elapsed_ms=elapsed,replica_files=[r['path'] for r in outputs],
                        dispatch=dispatch,updated=time.time())
            write(folder/'runs'/f"{job['key']}.json",result)
            print(job['key'],round(metrics['mean_ttft_ms'],3),'ms',flush=True)
            return result

        calibration=read(ROOT/'artifacts/v1_capacity_design_20260923/traces/code_calibration_original.json')
        # Identical per-replica calibration streams, same ingress path as formal runs.
        calibration=[dict(r,id=f"cal:{r['id']}:{j}") for r in calibration for j in range(len(gpus))]
        # Uniform full calibration warmup for ALL extension groups; saved, excluded from A/A.
        execute(dict(key='cal_warmup',arm='srp',repeat=0,phase='warmup',source='code'),calibration)
        aa=[]
        for repeat in (0,1):
            job=dict(key=f'aa_r{repeat}',arm='srp',repeat=repeat,phase='calibration',source='code')
            aa.append(execute(job,calibration))
        ratios=[]
        for key in ('mean_ttft_ms',):
            xs=[r['metrics'][key] for r in aa];ratios.append((max(xs)-min(xs))/st.fmean(xs))
        write(folder/'calibration.json',dict(passed=max(ratios)<=.05,relative_ranges=ratios))
        assert max(ratios)<=.05,'Calibration variation exceeds predeclared 5%; group excluded'
        for case in task['cases']:
            for load in task['loads']:
                reqs=requests_for(case,len(gpus),load)
                order=list(ARMS);random.Random(object_digest([taskkey,case,load])).shuffle(order)
                for repeat in range(design['repeats']):
                    for arm in order if repeat==0 else reversed(order):
                        job=dict(key=f'{case}_{load}_{arm}_r{repeat}',case=case,load=load,arm=arm,
                                 repeat=repeat,source='conversation',phase='compare')
                        execute(job,reqs)
        write(folder/'status.json',dict(state='complete',gpus=gpus,updated=time.time()))
    except BaseException:
        write(folder/'status.json',dict(state='failed',gpus=gpus,error=traceback.format_exc(),updated=time.time()))
        raise
    finally:
        for conn,process in zip(connections,processes):
            if process.is_alive():
                try:conn.send(dict(op='stop'))
                except Exception:pass
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():process.terminate();process.join(timeout=10)
            if process.is_alive():process.kill();process.join(timeout=5)
        for q in queues:q.close()
        for lock in locks:lock.close()


def initialize():
    design=read(OUT/'design.json')
    check=read(OUT/'cpu_validation.json')
    assert check['passed'] and check['sources']==sources()
    assert not (OUT/'manifest.json').exists()
    hardware=infra.hardware();gpus=design['gpu_preference']
    assert all('4090' in hardware[g]['name'] for g in gpus)
    affinity=infra.cpu_assign(gpus)
    used={c for cs in affinity.values() for c in cs}
    topology=[tuple(map(int,line.split(','))) for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True).splitlines() if not line.startswith('#')]
    used_cores={(core,socket) for cpu,core,socket in topology if cpu in used}
    spare=[];seen=set(used_cores)
    for cpu,core,socket in topology:
        if cpu in os.sched_getaffinity(0) and (core,socket) not in seen:
            spare.append(cpu);seen.add((core,socket))
    assert len(spare)>=len(gpus)
    manifest=dict(sources=sources(),design_sha256=digest(OUT/'design.json'),affinity=affinity,
        ingress_affinity={str(g):[spare[i]] for i,g in enumerate(gpus)},
        hardware={str(g):hardware[g] for g in gpus},
        versions={p:importlib.metadata.version(p) for p in ('vllm','torch','transformers')},created=time.time())
    assert manifest['versions']['vllm']=='0.9.2'
    write(OUT/'manifest.json',manifest)
    for relative in manifest['sources']:
        target=OUT/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--group',required=True);parser.add_argument('--gpus',required=True)
    args=parser.parse_args();group(args.group,[int(g) for g in args.gpus.split(',')])
