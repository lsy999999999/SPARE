"""Restartable three-GPU execution of the frozen capacity experiment."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.v1_budget_20260922 import run as ref

HERE = Path(__file__).resolve().parent
INPUT = ROOT/'artifacts/v1_capacity_design_20260923'
OUT = ROOT/'artifacts/v1_capacity_gpu_20260923_r1'
MODEL = Path(os.environ.get('A_V1_MODEL', 'models/Qwen2.5-7B-Instruct'))
read, write, digest = ref.read, ref.write, ref.digest
ref.OUT = OUT


def hw(gpu):
    raw = subprocess.check_output(['nvidia-smi',f'--id={gpu}',
        '--query-gpu=index,name,uuid,memory.total,driver_version','--format=csv,noheader,nounits'],text=True)
    index,name,uuid,memory,driver = [p.strip() for p in raw.strip().split(',')]
    assert 'RTX 4090' in name
    return dict(index=int(index), name=name, uuid=uuid, memory_mib=int(memory), driver=driver)


def hashes():
    files = list(HERE.glob('*.py')) + [HERE/'experiment.json',HERE/'PROTOCOL_CN.md']
    files += [ROOT/'research/v1_budget_20260922'/n for n in ('policy.py','scheduler.py','run.py')]
    result = {str(p.relative_to(ROOT)):digest(p) for p in files}
    dist = importlib.metadata.distribution('vllm')
    assert dist.version == '0.9.2'
    for n in ('v1/core/sched/scheduler.py','v1/core/kv_cache_manager.py',
              'v1/core/single_type_kv_cache_manager.py','v1/core/block_pool.py'):
        result['vllm/'+n] = digest(Path(dist.locate_file('vllm/'+n)))
    return result


def initialize():
    inputs = read(INPUT/'input_manifest.json')
    config = read(HERE/'experiment.json')
    assert config == inputs['config']
    assert not inputs['infeasible_jobs']
    assert read(INPUT/'local_audit.json')['complete']
    old = read(ROOT/'artifacts/v1_paper_comparison_multi_20260922_r3/manifest.json')
    for n in ('policy.py','scheduler.py','run.py'):
        name='research/v1_budget_20260922/'+n
        assert digest(ROOT/name)==old['frozen_reference_hashes'][name]
    m = dict(schema=1,implementation='capacity-native-v1-r1',config=config,
        inputs_sha256=digest(INPUT/'input_manifest.json'),local_audit_sha256=digest(INPUT/'local_audit.json'),
        source_hashes=hashes(), model=str(MODEL), model_revision=old['model_revision'],
        versions={p:importlib.metadata.version(p) for p in ('vllm','torch','transformers','numpy')},
        hardware={str(g):hw(g) for g in config['eligible_gpu_indices_at_preflight']},
        trace_hashes={p.name:digest(p) for p in (INPUT/'traces').glob('*.json')},
        expected=dict(smoke=27,aa=36,tune=96,compare=840),total_expected=999)
    path=OUT/'manifest.json'
    if path.exists():
        assert read(path)==m, 'Execution manifest changed; use a new run directory'
    else:
        write(path,m)
        for name in m['source_hashes']:
            if name.startswith('vllm/'):
                continue
            dest=OUT/'source'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes((ROOT/name).read_bytes())
        write(OUT/'input_manifest.json',inputs)
    return m


def jobs(m,phase,repeat=None):
    c=m['config']
    result=[]
    def append(key,gpu,cap,block,arrival,method,config,rep=0,stage=None,smoke_chunk=None):
        result.append(dict(key=key,phase=phase,stage=stage or phase,gpu=gpu,
            gpu_uuid=m['hardware'][str(gpu)]['uuid'],capacity=cap,block=block,
            arrival=arrival,method=method,config=config,repeat=rep,smoke_chunk=smoke_chunk))
    if phase=='smoke':
        methods=('srp_matched','a_frozen','fixed_small','occupancy',
                 'occupancy_finish_guard','a_no_finish_guard','a_any_positive_saving')
        for gpu in c['eligible_gpu_indices_at_preflight']:
            for method in methods:
                cfg=dict(c['base_config'])
                if method=='fixed_small': cfg.update(budget=512)
                append(f'smoke_g{gpu}_{method}',gpu,768,None,None,method,cfg)
            for chunk in (64,256):
                append(f'smoke_g{gpu}_native{chunk}',gpu,768,None,None,'native',
                       dict(budget=1024,chunk=chunk,concurrency=32),smoke_chunk=chunk)
    elif phase=='aa':
        for gpu in c['eligible_gpu_indices_at_preflight']:
            for cap in (768,7445):
                for source in ('code','conversation'):
                    for rep in range(3):
                        append(f'aa_g{gpu}_k{cap}_{source}_r{rep}',gpu,cap,source+'_calibration',
                            'burst20ms','srp_matched',dict(c['base_config']),rep)
    else:
        selection=read(OUT/'selection.json') if phase=='compare' else None
        inputs=read(INPUT/'input_manifest.json')
        for j in inputs['jobs']:
            if (j['stage']=='tune') != (phase=='tune'): continue
            if phase=='compare' and repeat is not None and j['repeat']!=repeat: continue
            method=j['method']
            cfg=dict(c['base_config'])
            if method in ('fixed_small','fixed_small_dev'):
                budget=j['candidate'] if phase=='tune' else selection['fixed_small_budget']
                cfg.update(budget=budget,chunk=min(budget,256))
                method='fixed_small'
            if method in ('occupancy','occupancy_dev','occupancy_finish_guard'):
                cfg['occupancy_threshold']=j['candidate'] if phase=='tune' else selection['occupancy_threshold']
                if method=='occupancy_dev': method='occupancy'
            append(j['id'],j['gpu'],j['capacity'],j['block'],j['arrival'],method,cfg,
                   j['repeat'],j['stage'])
    # Stable method rotation within each paired cell; second repeat reverses it.
    grouped={}
    for j in result:
        group=(j['gpu'],j['capacity'],j['block'] or '',j['arrival'] or '',j['repeat'])
        grouped.setdefault(group,[]).append(j)
    ordered=[]
    for gi,(group,arms) in enumerate(sorted(grouped.items())):
        arms.sort(key=lambda j:(j['method'],str(j['config'])))
        shift=(gi+group[-1])%len(arms)
        arms=arms[shift:]+arms[:shift]
        if group[-1]%2: arms.reverse()
        ordered.extend(arms)
    return ordered


def requests(m,j):
    if j['phase']=='smoke':
        if j['smoke_chunk']:
            return [dict(id='smoke_single',prompt=1024,output=16,arrival_ms=0)]
        return [dict(id=f'smoke{i}',prompt=256+128*i,output=24,arrival_ms=0) for i in range(8)]
    name=f"{j['block']}_{j['arrival']}.json"
    assert digest(INPUT/'traces'/name)==m['trace_hashes'][name]
    return read(INPUT/'traces'/name)


def audit(row):
    if row['status']=='complete':
        ref.audit_result(row)
    else:
        assert row['error']=='TimeoutError: Predeclared episode deadline exceeded',row['error']
        ref.audit_result(dict(row,job=dict(row['job'],phase='tune')),allow_timeout=True)


def build_engine(m,cap,gpu):
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    assert str(torch.cuda.get_device_properties(0).uuid).lower()==m['hardware'][str(gpu)]['uuid'].removeprefix('GPU-').lower()
    engine=LLMEngine.from_engine_args(EngineArgs(model=str(MODEL),dtype='half',
        max_model_len=12288,max_num_seqs=64,max_num_batched_tokens=2048,
        enable_chunked_prefill=True,enable_prefix_caching=False,block_size=16,
        gpu_memory_utilization=.9,enforce_eager=True,disable_log_stats=True,seed=0,
        num_gpu_blocks_override=cap,
        scheduler_cls='research.v1_capacity_ablation_20260923.scheduler.CapacityScheduler'))
    scheduler=engine.engine_core.engine_core.scheduler
    assert scheduler.kv_cache_config.num_blocks==cap
    warm=ref.episode(engine,scheduler,[dict(id=f'warm{i}',prompt=256,output=16,arrival_ms=0)
        for i in range(4)],'srp_matched',m['config']['base_config'])
    assert warm['status']=='complete',warm['error']
    return engine,scheduler


def worker(m,phase,gpu,cap,repeat):
    assert hashes()==m['source_hashes']
    assert hw(gpu)==m['hardware'][str(gpu)]
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    assert sorted(os.sched_getaffinity(0))==m['config']['proposed_cpu_affinity'][str(gpu)]
    pending=[]
    for j in jobs(m,phase,repeat):
        if j['gpu']!=gpu or j['capacity']!=cap: continue
        path=OUT/'runs'/(j['key']+'.json')
        if path.exists():
            row=read(path)
            assert row['job']==j and row['requests']==requests(m,j)
            audit(row)
        else: pending.append(j)
    if not pending: return
    engine,scheduler=build_engine(m,cap,gpu)
    for j in pending:
        write(OUT/f'worker_g{gpu}.json',dict(state='running',job=j,pid=os.getpid(),updated=time.time()))
        row=ref.episode(engine,scheduler,requests(m,j),j['method'],j['config'],timeout=300)
        rounds=row.pop('rounds')
        for metric in ('budget_changed','rejected_guard','rejected_saving','rejected_infeasible'):
            row['metrics'][metric]=sum(r[metric] for r in rounds)
        e2e=[c['finish_ms']-c['arrival_ms'] for c in row['clocks'].values() if c['finish_ms'] is not None]
        row['metrics']['mean_e2e_ms']=st.fmean(e2e) if row['status']=='complete' else None
        row['metrics']['p95_e2e_ms']=ref.quantile(e2e,.95) if row['status']=='complete' else None
        trace=OUT/'rounds'/(j['key']+'.json.gz')
        trace.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(trace,'wt',compresslevel=1) as f: json.dump(rounds,f)
        row.update(job=j,rounds_path=str(trace.relative_to(OUT)),rounds_sha256=digest(trace),
                   recorded_at=time.time())
        write(OUT/'runs'/(j['key']+'.json'),row)
        audit(row)
        if j['smoke_chunk']:
            allocation=rounds[0]['allocations'][0]
            assert allocation['tokens']==j['smoke_chunk']
            assert allocation['blocks_after']==j['smoke_chunk']//16
        print(j['key'],j['method'],row['status'],row['metrics']['mean_ttft_ms'],flush=True)
        write(OUT/f'worker_g{gpu}.json',dict(state='saved',job=j,pid=os.getpid(),
            status=row['status'],updated=time.time()))
        if row['status']!='complete': raise SystemExit(75)


def run_gpu(m,phase,gpu,repeat):
    planned=[j for j in jobs(m,phase,repeat) if j['gpu']==gpu]
    for cap in sorted({j['capacity'] for j in planned}):
        cell=[j for j in planned if j['capacity']==cap]
        attempts=0
        while any(not (OUT/'runs'/(j['key']+'.json')).exists() for j in cell):
            used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used>512: raise RuntimeError(f'GPU {gpu} occupied ({used} MiB); no unrelated process will be stopped')
            affinity=','.join(map(str,m['config']['proposed_cpu_affinity'][str(gpu)]))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=OUT/'logs'/f'{phase}_g{gpu}_k{cap}_r{repeat}.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            command=['taskset','-c',affinity,sys.executable,'-B','-u',str(HERE/'run.py'),
                     '--phase',phase,'--worker',str(gpu),'--capacity',str(cap)]
            if repeat is not None: command += ['--repeat',str(repeat)]
            with log.open('a') as f:
                code=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,
                                    timeout=len(cell)*330+900).returncode
            if code not in (0,75): raise RuntimeError(f'Worker failed ({code}): {log}')
            attempts+=1
            if attempts>len(cell)+1: raise RuntimeError('No forward progress')


def check_aa(m):
    groups={}
    for j in jobs(m,'aa'):
        row=read(OUT/'runs'/(j['key']+'.json')); audit(row)
        assert row['status']=='complete'
        groups.setdefault((j['gpu'],j['capacity'],j['block']),[]).append(row)
    report=[]
    for group,rows in groups.items():
        assert len(rows)==3
        values={name:[r['elapsed_ms'] if name=='elapsed_ms' else r['metrics'][name] for r in rows]
                for name in ('mean_ttft_ms','elapsed_ms')}
        ratios={name:(max(xs)-min(xs))/st.fmean(xs) for name,xs in values.items()}
        report.append(dict(group=group,range_over_mean=ratios,passed=max(ratios.values())<=.05))
    write(OUT/'aa_report.json',dict(passed=all(r['passed'] for r in report),groups=report))
    if not all(r['passed'] for r in report): raise RuntimeError('A/A exceeds predeclared 5% gate; see aa_report.json')


def select_baselines(m):
    rows=[read(OUT/'runs'/(j['key']+'.json')) for j in jobs(m,'tune')]
    refs={(r['job']['block'],r['job']['capacity'],r['job']['arrival']):r['metrics']['mean_ttft_ms']
          for r in rows if r['job']['method']=='fixed_small' and r['job']['config']['budget']==1024}
    assert len(refs)==12 and all(v is not None and v>0 for v in refs.values()),'Development reference incomplete'
    selections={}
    scores={}
    for method,param,candidates in [('fixed_small','budget',m['config']['fixed_small_candidates']),
            ('occupancy','occupancy_threshold',m['config']['occupancy_threshold_candidates'])]:
        scored=[]
        for candidate in candidates:
            subset=[r for r in rows if r['job']['method']==method and r['job']['config'][param]==candidate]
            assert len(subset)==12
            score=st.fmean(r['metrics']['mean_ttft_ms']/refs[(r['job']['block'],r['job']['capacity'],r['job']['arrival'])]
                          for r in subset) if all(r['status']=='complete' for r in subset) else None
            scored.append(dict(candidate=candidate,score=score))
        eligible=[s for s in scored if s['score'] is not None]
        assert eligible, f'No complete development candidate: {method}'
        selections['fixed_small_budget' if method=='fixed_small' else 'occupancy_threshold']=min(
            eligible,key=lambda s:(s['score'],-s['candidate']))['candidate']
        scores[method]=scored
    selections.update(scores=scores,tuning_hashes={r['job']['key']:digest(OUT/'runs'/(r['job']['key']+'.json')) for r in rows})
    path=OUT/'selection.json'
    if path.exists(): assert read(path)==selections
    else: write(path,selections)


def status():
    counts={}
    for path in (OUT/'runs').glob('*.json'):
        row=read(path)
        key=row['job']['phase']+'/'+row['status']
        counts[key]=counts.get(key,0)+1
    state=read(OUT/'status.json') if (OUT/'status.json').exists() else dict(state='not_started')
    workers={p.stem:read(p) for p in OUT.glob('worker_g*.json')}
    print(json.dumps(dict(controller=state,counts=counts,saved=sum(counts.values()),expected=999,
                         workers=workers),indent=2))


def summarize(m):
    rows=[]
    missing=[]
    for phase in ('smoke','aa','tune','compare'):
        for j in jobs(m,phase):
            path=OUT/'runs'/(j['key']+'.json')
            if not path.exists(): missing.append(j['key']); continue
            row=read(path); audit(row); rows.append(row)
    groups={}
    for row in rows:
        j=row['job']
        if j['phase']!='compare': continue
        key=f"k{j['capacity']}/{j['arrival']}/{j['method']}"
        groups.setdefault(key,[]).append(row)
    summary={}
    for key,rs in groups.items():
        complete=all(r['status']=='complete' for r in rs)
        summary[key]=dict(runs=len(rs),complete=complete,
            metrics={metric:st.fmean(r['metrics'][metric] for r in rs) if complete else None
                     for metric in ('mean_ttft_ms','p95_ttft_ms','mean_e2e_ms','throughput_tokens_s',
                         'preemptions','recomputed_tokens','interventions','decision_ms_mean')})
    payload=dict(complete=not missing and len(rows)==999,all_runs_successful=all(r['status']=='complete' for r in rows),
                 runs=len(rows),missing=missing,groups=summary,
                 failures=[r['job']['key'] for r in rows if r['status']!='complete'])
    write(OUT/'summary.json',payload)
    lines=['# Capacity study GPU results','',f"Saved {len(rows)}/999; failed {len(payload['failures'])}.",'',
           '| Capacity / arrivals / method | Runs | Mean TTFT ms | Output tokens/s | Preemptions |',
           '| --- | ---: | ---: | ---: | ---: |']
    for key,g in sorted(summary.items()):
        met=g['metrics']
        lines.append(f"| {key} | {g['runs']} | {met['mean_ttft_ms']} | {met['throughput_tokens_s']} | {met['preemptions']} |")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('all','smoke','aa','tune','compare','status','summary'),default='all')
    p.add_argument('--worker',type=int)
    p.add_argument('--capacity',type=int)
    p.add_argument('--repeat',type=int)
    args=p.parse_args()
    if args.phase=='status': status(); return
    OUT.mkdir(parents=True,exist_ok=True)
    if args.worker is not None:
        worker(read(OUT/'manifest.json'),args.phase,args.worker,args.capacity,args.repeat); return
    if args.phase=='summary': summarize(read(OUT/'manifest.json')); return
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=initialize()
        locks=[]
        for gpu in m['config']['eligible_gpu_indices_at_preflight']:
            gpu_lock=open(f'/tmp/query-capacity-gpu{gpu}.lock','a')
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(gpu_lock)
        try:
            phases=('smoke','aa','tune','compare') if args.phase=='all' else (args.phase,)
            for phase in phases:
                for rep in ((0,1) if phase=='compare' else (None,)):
                    write(OUT/'status.json',dict(state='running',phase=phase,repeat=rep,pid=os.getpid(),updated=time.time()))
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        futures=[pool.submit(run_gpu,m,phase,gpu,rep) for gpu in m['config']['eligible_gpu_indices_at_preflight']]
                        for future in futures: future.result()
                if phase=='aa': check_aa(m)
                if phase=='tune': select_baselines(m)
            if args.phase=='all': summarize(m)
            write(OUT/'status.json',dict(state='finished',phase=args.phase,pid=os.getpid(),updated=time.time()))
        except Exception as exc:
            write(OUT/'status.json',dict(state='failed',error=str(exc),pid=os.getpid(),updated=time.time()))
            raise


if __name__=='__main__':
    main()
