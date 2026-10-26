"""Restartable detached pipeline: native validation, CPU simulation, paired GPU pilot."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import gzip
import importlib.metadata
import os
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_state_aware_20260923.common import *
from research.v1_budget_20260922 import run as ref
ref.OUT=OUT
MODEL='models/Qwen2.5-7B-Instruct'


def hardware():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.used,utilization.gpu,driver_version',
                                 '--format=csv,noheader,nounits'],text=True)
    result={}
    for line in raw.strip().splitlines():
        i,name,uuid,used,util,driver=[x.strip() for x in line.split(',')]
        result[int(i)]=dict(index=int(i),name=name,uuid=uuid,used=int(used),util=int(util),driver=driver)
    return result


def compute_pids(uuid):
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return {int(p.strip()) for line in raw.splitlines() if ',' in line
            for u,p in [line.split(',',1)] if u.strip()==uuid}


def idle(h): return h['used']<256 and h['util']==0 and not compute_pids(h['uuid'])


def acquire(path):
    f=open(path,'a')
    try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except Exception: f.close(); raise
    return f


def cpu_assign(gpus):
    # Use distinct physical cores, prioritizing sampled idle time.
    def sample():
        ans={}
        for line in Path('/proc/stat').read_text().splitlines():
            fields=line.split()
            if fields and fields[0][3:].isdigit():
                vals=list(map(int,fields[1:])); ans[int(fields[0][3:])]=(sum(vals[:8]),vals[3]+vals[4])
        return ans
    first=sample(); time.sleep(1); second=sample()
    allowed=os.sched_getaffinity(0)
    cores=[]; seen=set()
    raw=subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True)
    for line in raw.splitlines():
        if line.startswith('#'): continue
        cpu,core,socket=map(int,line.split(',')); ident=(socket,core)
        if cpu in allowed and ident not in seen: seen.add(ident); cores.append(cpu)
    cores.sort(key=lambda c:-(second[c][1]-first[c][1])/max(1,second[c][0]-first[c][0]))
    assert len(cores)>=3*len(gpus)
    return {str(g):sorted(cores[i*3:i*3+3]) for i,g in enumerate(gpus)}


def initialize():
    path=OUT/'manifest.json'
    if path.exists(): return check_manifest()
    h=hardware(); gpus=[g for g,v in h.items() if 'RTX 4090' in v['name'] and idle(v)]
    assert gpus,'No idle RTX 4090 available'
    # Six paired cells bound useful concurrency for this pilot.
    gpus=gpus[:len(BLOCKS)*len(CAPS)]
    cells=[dict(key=f'{block}_k{cap}',block=block,capacity=cap,gpu=gpus[i%len(gpus)])
           for i,(cap,block) in enumerate((c,b) for c in CAPS for b in BLOCKS)]
    old=read(OLD/'manifest.json')
    m=dict(schema=1,claim='development pilot; two reused windows; no independent confirmation',
        source_hashes=sources(),trace_hashes={b+'_original.json':digest(INPUT/(b+'_original.json'))
                                          for b in BLOCKS+('code_calibration',)},
        hardware={str(g):{k:v for k,v in h[g].items() if k not in ('used','util')} for g in gpus},
        gpus=gpus,cpu_affinity=cpu_assign(gpus),cells=cells,methods=list(METHODS),
        repeats=2,compare_expected=72,validation_expected=9*len(gpus),model=MODEL,
        model_revision=old['model_revision'],versions={p:importlib.metadata.version(p) for p in ('vllm','torch','transformers','numpy')},
        configs={method:config(method) for method in METHODS},deadline_s=300)
    assert m['versions']['vllm']=='0.9.2'
    dist=importlib.metadata.distribution('vllm')
    m['native_hashes']={n:digest(dist.locate_file('vllm/'+n)) for n in
        ('v1/core/sched/scheduler.py','v1/core/block_pool.py','v1/core/kv_cache_manager.py','v1/core/single_type_kv_cache_manager.py')}
    for name in m['source_hashes']:
        dest=OUT/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    write(path,m)
    return m


def jobs(m,cell):
    g=cell['gpu']; result=[]
    if cell['key'].startswith('validation_'):
        for method in METHODS:
            result.append(dict(key=f'g{g}_smoke_{method}',phase='smoke',method=method,config=config(method),repeat=0))
        for rep in range(3):
            result.append(dict(key=f'g{g}_aa_r{rep}',phase='aa',method='srp_matched',config=dict(CONFIG),repeat=rep))
    else:
        shift=m['cells'].index(cell)%len(METHODS)
        order=list(METHODS[shift:]+METHODS[:shift])
        for rep in range(2):
            for method in (order if rep==0 else list(reversed(order))):
                result.append(dict(key=f"{cell['key']}_{method}_r{rep}",phase='compare',method=method,
                                   config=m['configs'][method],repeat=rep))
    for j in result: j.update(cell=cell,gpu_uuid=m['hardware'][str(g)]['uuid'])
    return result


def requests(j):
    if j['phase']=='smoke':
        return [dict(id=f'smoke{i}',prompt=256+128*i,output=24,arrival_ms=0) for i in range(8)]
    block='code_calibration' if j['phase']=='aa' else j['cell']['block']
    return read(INPUT/(block+'_original.json'))


def audit(row):
    if row['status']=='complete': ref.audit_result(row)
    else:
        assert row['error']=='TimeoutError: Predeclared episode deadline exceeded',row['error']
        # The historical verifier accepts timeout records only under a tune tag.
        ref.audit_result(dict(row,job=dict(row['job'],phase='tune')),allow_timeout=True)


def worker(cellkey):
    m=check_manifest()
    cell=next((c for c in m['cells'] if c['key']==cellkey),None)
    if cell is None:
        g=int(cellkey.split('_g')[1]); cell=dict(key=cellkey,gpu=g,capacity=768,block=None)
    g=cell['gpu']; expected=m['hardware'][str(g)]
    actual_hw=hardware()[g]
    assert all(actual_hw[k]==v for k,v in expected.items())
    dist=importlib.metadata.distribution('vllm')
    assert {n:digest(dist.locate_file('vllm/'+n)) for n in m['native_hashes']}==m['native_hashes']
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(g)
    assert sorted(os.sched_getaffinity(0))==m['cpu_affinity'][str(g)]
    pending=[]
    for j in jobs(m,cell):
        path=OUT/'runs'/(j['key']+'.json')
        if path.exists():
            row=read(path); assert row['job']==j and row['requests']==requests(j); audit(row)
        else: pending.append(j)
    if not pending: return
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    assert str(torch.cuda.get_device_properties(0).uuid).lower()==expected['uuid'].removeprefix('GPU-').lower()
    engine=LLMEngine.from_engine_args(EngineArgs(model=MODEL,dtype='half',max_model_len=12288,
        max_num_seqs=64,max_num_batched_tokens=2048,enable_chunked_prefill=True,
        enable_prefix_caching=False,block_size=16,gpu_memory_utilization=.9,
        enforce_eager=True,disable_log_stats=True,seed=0,num_gpu_blocks_override=cell['capacity'],
        scheduler_cls='research.v1_state_aware_20260923.scheduler.StateScheduler'))
    scheduler=engine.engine_core.engine_core.scheduler
    assert scheduler.kv_cache_config.num_blocks==cell['capacity']
    warm=ref.episode(engine,scheduler,[dict(id=f'warm{i}',prompt=256,output=16,arrival_ms=0) for i in range(4)],'srp_matched',dict(CONFIG))
    assert warm['status']=='complete',warm['error']
    for j in pending:
        state(f'worker_g{g}',state='running',job=j)
        row=ref.episode(engine,scheduler,requests(j),j['method'],j['config'],timeout=300)
        rounds=row.pop('rounds')
        for field in ('rejected_guard','rejected_saving','rejected_infeasible','pressure_triggered'):
            row['metrics'][field]=sum(r[field] for r in rounds)
        row['metrics']['max_consecutive_reductions']=max((r['max_consecutive_reductions'] for r in rounds),default=0)
        es=[c['finish_ms']-c['arrival_ms'] for c in row['clocks'].values() if c['finish_ms'] is not None]
        row['metrics']['mean_e2e_ms']=st.fmean(es) if row['status']=='complete' else None
        row['metrics']['p95_e2e_ms']=ref.quantile(es,.95) if row['status']=='complete' else None
        trace=OUT/'rounds'/(j['key']+'.json.gz'); trace.parent.mkdir(parents=True,exist_ok=True)
        temp=trace.with_name(trace.name+'.tmp')
        with gzip.open(temp,'wt',compresslevel=1) as f: __import__('json').dump(rounds,f)
        temp.replace(trace)
        row.update(job=j,rounds_path=str(trace.relative_to(OUT)),rounds_sha256=digest(trace),recorded_at=time.time())
        # Validate before making a completed run visible to the controller.
        try:
            audit(row)
        except Exception:
            write(OUT/'failures'/(j['key']+'.json'),row)
            raise
        write(OUT/'runs'/(j['key']+'.json'),row)
        state(f'worker_g{g}',state='saved',job=j,result=row['status'])
        print(j['key'],row['status'],row['metrics']['mean_ttft_ms'],flush=True)
        if row['status']!='complete': raise SystemExit(75)


def quarantine(m,cell,reason):
    dest=OUT/'quarantine'/f"{cell['key']}_{time.time_ns()}"
    for j in jobs(m,cell):
        for folder,suffix in [('runs','.json'),('rounds','.json.gz')]:
            path=OUT/folder/(j['key']+suffix)
            if path.exists():
                (dest/folder).mkdir(parents=True,exist_ok=True); shutil.move(str(path),str(dest/folder/path.name))
    write(dest/'reason.json',dict(reason=reason,time=time.time()))


def run_cell(m,cell):
    g=cell['gpu']; uuid=m['hardware'][str(g)]['uuid']
    while True:
        missing=[j for j in jobs(m,cell) if not (OUT/'runs'/(j['key']+'.json')).exists()]
        if not missing: break
        while not idle(hardware()[g]):
            state(f'worker_g{g}',state='waiting_for_idle_gpu',cell=cell)
            time.sleep(10)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),CUDA_DEVICE_ORDER='PCI_BUS_ID',
            VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=OUT/'logs'/(cell['key']+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
        cmd=['taskset','-c',','.join(map(str,m['cpu_affinity'][str(g)])),sys.executable,'-B','-u',str(HERE/'run.py'),'--worker',cell['key']]
        contaminated=False; start=time.monotonic()
        with log.open('a') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    external=compute_pids(uuid)-{proc.pid}
                    if external:
                        contaminated=True; proc.terminate()
                        try: proc.wait(timeout=15)
                        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                        quarantine(m,cell,dict(external_gpu_pids=sorted(external)))
                        break
                    if time.monotonic()-start>900+330*len(missing):
                        proc.terminate()
                        try: proc.wait(timeout=15)
                        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                        raise RuntimeError(f'Worker watchdog exceeded for {cell}')
                    time.sleep(3)
                code=proc.wait()
            finally:
                if proc.poll() is None: proc.terminate(); proc.wait(timeout=15)
        if contaminated: continue
        if code not in (0,75): raise RuntimeError(f'Worker {cell["key"]} exited {code}; see {log}')
    for j in jobs(m,cell):
        row=read(OUT/'runs'/(j['key']+'.json')); assert row['job']==j; audit(row)
    state(f'worker_g{g}',state='cell_finished',cell=cell)


def validate_gpu(m,g):
    cell=dict(key=f'validation_g{g}',gpu=g,capacity=768,block=None)
    run_cell(m,cell)
    rows=[read(OUT/'runs'/(j['key']+'.json')) for j in jobs(m,cell)]
    assert all(r['status']=='complete' for r in rows),'Validation incomplete'
    aa=[r for r in rows if r['job']['phase']=='aa']
    ratios={metric:(max(vals)-min(vals))/st.fmean(vals) for metric,vals in
            [('mean_ttft_ms',[r['metrics']['mean_ttft_ms'] for r in aa]),('elapsed_ms',[r['elapsed_ms'] for r in aa])]}
    write(OUT/f'aa_g{g}.json',dict(ratios=ratios,passed=max(ratios.values())<=.05))
    assert max(ratios.values())<=.05,f'GPU {g} A/A noise above 5%: {ratios}'


def controller():
    OUT.mkdir(parents=True,exist_ok=True)
    master=acquire(OUT/'controller.lock'); locks=[]
    try:
        m=initialize()
        for g in m['gpus']:
            locks.append(acquire(f'/tmp/query-capacity-gpu{g}.lock'))
        env=dict(os.environ,VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',
                 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        for label,command in [('validation',[sys.executable,'-B','-m','unittest',
                'research.v1_state_aware_20260923.test_policy','research.v1_state_aware_20260923.test_native','-v']),
                ('simulation',[sys.executable,'-B','-u',str(HERE/'simulate.py')])]:
            if (OUT/(label+'_done.json')).exists(): continue
            state('status',state='running',phase=label)
            with (OUT/(label+'.log')).open('a') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            write(OUT/(label+'_done.json'),dict(complete=True,source_hashes=m['source_hashes']))
        state('status',state='running',phase='gpu_validation')
        with ThreadPoolExecutor(max_workers=len(m['gpus'])) as pool:
            futures=[pool.submit(validate_gpu,m,g) for g in m['gpus']]
            for future in futures: future.result()
        state('status',state='running',phase='gpu_compare')
        def run_gpu(g):
            for cell in m['cells']:
                if cell['gpu']==g: run_cell(m,cell)
        with ThreadPoolExecutor(max_workers=len(m['gpus'])) as pool:
            futures=[pool.submit(run_gpu,g) for g in m['gpus']]
            for future in futures: future.result()
        state('status',state='running',phase='audit_summary')
        subprocess.run([sys.executable,'-B',str(HERE/'report.py')],cwd=ROOT,check=True)
        state('status',state='finished',phase='complete')
    except Exception as exc:
        state('status',state='failed',error=str(exc),traceback=traceback.format_exc()); raise
    finally:
        for lock in locks: lock.close()
        master.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--worker'); args=p.parse_args()
    if args.worker: worker(args.worker)
    else: controller()
