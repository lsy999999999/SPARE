"""Bounded detached matrix, idle-only GPUs, whole-group provenance and recovery."""
import argparse,gzip,importlib.metadata,json,os,random,signal,statistics as st,subprocess,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_arrival_kv_papers_20260925.common import *
from research.v1_state_aware_20260923 import run as infra
from research.v1_budget_20260922 import run as ref
ref.OUT=OUT

def save(engine,scheduler,j,g,folder,deadline):
    target=folder/'runs'/f"{j['key']}.json"
    assert not target.exists()
    reqs=j.get('requests') or read(OUT/f"inputs/{j['case']}.json")
    left=deadline-time.time()
    if left<=0:raise TimeoutError('Global deadline reached before episode')
    write(OUT/f'worker_g{g}.json',dict(state='running',job=j['key'],group=j.get('group'),updated=time.time()))
    context(scheduler,j.get('source','code'),j['repeat'])
    result=ref.episode(engine,scheduler,reqs,j['method'],j['config'],timeout=min(read(OUT/'design.json')['episode_timeout_s'],left))
    rounds=result.pop('rounds');path=folder/'rounds'/f"{j['key']}.json.gz";path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',compresslevel=1) as stream:json.dump(rounds,stream,separators=(',',':'))
    result['metrics']['mean_e2e_ms']=st.fmean(c['finish_ms']-c['arrival_ms'] for c in result['clocks'].values()) if result['status']=='complete' else None
    result.update(job=j,gpu=g,rounds_path=str(path.relative_to(OUT)),rounds_sha256=digest(path),saved_at=time.time())
    write(target,result)
    assert result['status']=='complete',result['error']
    ref.audit_result(result)
    print(j['key'],round(result['metrics']['mean_ttft_ms'],2),flush=True)
    return result

def worker(taskpath,g):
    lock=infra.acquire(f'/tmp/query-capacity-gpu{g}.lock')
    m=read(OUT/'manifest.json');assert m['sources']==source_hashes()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(g)
    assert sorted(os.sched_getaffinity(0))==m['affinity'][str(g)]
    dist=importlib.metadata.distribution('vllm')
    assert {n:digest(dist.locate_file('vllm/'+n)) for n in m['native_hashes']}==m['native_hashes']
    assert digest(OUT/'histories.json')==read(OUT/'design.json')['histories_sha256']
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    assert str(torch.cuda.get_device_properties(0).uuid).lower()==m['hardware'][str(g)]['uuid'].removeprefix('GPU-').lower()
    task=read(taskpath);cap=task['cell']['capacity'];deadline=read(OUT/'launch.json')['deadline']
    engine=LLMEngine.from_engine_args(EngineArgs(model=MODEL,dtype='half',max_model_len=8176,
        max_num_seqs=64,max_num_batched_tokens=2048,enable_chunked_prefill=True,enable_prefix_caching=False,block_size=16,
        gpu_memory_utilization=.9,enforce_eager=True,disable_log_stats=True,seed=0,num_gpu_blocks_override=cap,
        scheduler_cls='research.v1_arrival_kv_papers_20260925.scheduler.MatrixScheduler'))
    scheduler=engine.engine_core.engine_core.scheduler
    warm=[dict(id=f'w{i}',prompt=256,output=16,arrival_ms=0) for i in range(4)]
    for _ in range(2):assert ref.episode(engine,scheduler,warm,'threshold',dict(BASE,threshold='off'),timeout=60)['status']=='complete'
    calibration=OUT/f'calibration/g{g}_c{cap}'
    if not (calibration/'status.json').exists():
        # Existing incomplete calibration is never silently overwritten or counted as passed.
        assert not (calibration/'runs').exists(),'Incomplete calibration requires review'
        cal=read(ROOT/'artifacts/v1_capacity_design_20260923/traces/code_calibration_original.json');results=[]
        for rep in (0,1):
            j=dict(key=f'aa_g{g}_c{cap}_r{rep}',phase='aa',capacity=cap,repeat=rep,method='threshold',config=dict(BASE,threshold='off'),requests=cal)
            results.append(save(engine,scheduler,j,g,calibration,deadline))
        ratios=[(max(xs)-min(xs))/st.fmean(xs) for xs in ([r['metrics']['mean_ttft_ms'] for r in results],[r['elapsed_ms'] for r in results])]
        write(calibration/'status.json',dict(passed=max(ratios)<=.05,ratios=ratios,gpu=g,capacity=cap))
    assert read(calibration/'status.json')['passed'],'A/A calibration failed'
    folder=OUT/task['folder']
    for j in task['jobs']:save(engine,scheduler,j,g,folder,deadline)
    write(folder/'worker_complete.json',dict(passed=True,updated=time.time()))
    lock.close()

def build_tasks(d):
    tasks=[]
    for cell in d['cells']:
        group=f"{cell['case']}_c{cell['capacity']}";jobs=[]
        rng=random.Random(int(object_digest(cell)[:12],16))
        order=list(d['methods']);rng.shuffle(order)
        for rep in range(d['repeats']):
            # Two shuffled/reverse pairs reduce drift; randomization is outcome-independent.
            if rep==2:rng.shuffle(order)
            for arm in (order if rep%2==0 else list(reversed(order))):
                jobs.append(dict(key=f'{group}_{arm}_r{rep}',group=group,**cell,phase='compare',arm=arm,repeat=rep,
                                 method=d['arms'][arm]['method'],config=d['arms'][arm]['config']))
        tasks.append(dict(group=group,cell=cell,jobs=jobs,attempt=0))
    # Interleave provenance, capacities and stages before any outcome is known.
    random.Random(2026092512).shuffle(tasks)
    return tasks

def controller():
    lock=infra.acquire(OUT/'controller.lock');d=read(OUT/'design.json');v=read(OUT/'cpu_validation.json')
    assert v['passed'] and v['sources']==source_hashes() and v['design_sha256']==digest(OUT/'design.json')
    for name,h in d['trace_hashes'].items():assert object_digest(read(OUT/f'inputs/{name}.json'))==h
    assert not (OUT/'manifest.json').exists(),'Do not restart controller over existing evidence'
    assert digest(OUT/'histories.json')==d['histories_sha256']
    h=infra.hardware();gs=[g for g in d['gpu_preference'] if g in h and '4090' in h[g]['name']]
    assert gs
    affinity=infra.cpu_assign(gs)
    cpu_used={c for group in affinity.values() for c in group}
    # Exclude SMT siblings of GPU-driver cores from simulator CPU selection.
    topology=[tuple(map(int,line.split(','))) for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True).splitlines() if not line.startswith('#')]
    used_physical={(core,socket) for cpu,core,socket in topology if cpu in cpu_used}
    spare=[cpu for cpu,core,socket in topology if cpu in os.sched_getaffinity(0) and (core,socket) not in used_physical]
    assert spare,'Need separate CPU core for simulator'
    m=dict(gpus=gs,hardware={str(g):h[g] for g in gs},affinity=affinity,simulator_affinity=spare[:2],
        sources=source_hashes(),design_sha256=digest(OUT/'design.json'),model=MODEL,created=time.time(),
        versions={p:importlib.metadata.version(p) for p in ('vllm','torch','transformers')},
        tasks=build_tasks(d))
    assert m['versions']['vllm']=='0.9.2'
    dist=importlib.metadata.distribution('vllm')
    m['native_hashes']={name:digest(dist.locate_file('vllm/'+name)) for name in ('v1/core/sched/scheduler.py','v1/core/block_pool.py','v1/core/kv_cache_manager.py','v1/core/single_type_kv_cache_manager.py')}
    m['supporting_files']={str(p.relative_to(ROOT)):digest(p) for p in list((HERE/'analysis').glob('*.py'))+list(HERE.glob('*.md'))}
    write(OUT/'manifest.json',m)
    for name in m['supporting_files']:
        dest=OUT/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    for name in m['sources']:
        dest=OUT/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    # Simulator shares no CPU affinity with GPU drivers. It validates states, not GPU time.
    simlog=(OUT/'simulator.log').open('a')
    sim=subprocess.Popen(['taskset','-c',','.join(map(str,m['simulator_affinity'])),sys.executable,'-B','-u',str(HERE/'validate.py'),'--full'],
        cwd=ROOT,stdin=subprocess.DEVNULL,stdout=simlog,stderr=subprocess.STDOUT)
    deadline=read(OUT/'launch.json')['deadline'];pending=list(m['tasks']);mutex=threading.Lock();counts=dict(complete=0,failed=0,retried=0)
    active={};disabled=set();finished=set()
    def status():
        write(OUT/'status.json',dict(state='running',updated=time.time(),deadline=deadline,planned_groups=len(m['tasks']),
            pending=len(pending),active=active,counts=counts,disabled_gpu_capacity=sorted(disabled)))
    def consume(g):
        while time.time()<deadline:
            with mutex:
                if not pending:return
            import shutil
            if shutil.disk_usage(OUT).free<15*1024**3:raise RuntimeError('Disk reserve below 15 GiB; stopping with evidence preserved')
            if not infra.idle(infra.hardware()[g]):
                write(OUT/f'worker_g{g}.json',dict(state='waiting_for_idle_gpu',updated=time.time()));time.sleep(15);continue
            with mutex:
                idx=next((i for i,t in enumerate(pending) if f"{g}:{t['cell']['capacity']}" not in disabled),None)
                if idx is None:
                    if not pending:return
                    task=None
                else:
                    task=pending.pop(idx);active[str(g)]=task['group'];status()
            if task is None:time.sleep(15);continue
            group=task['group'];attempt=task['attempt'];task=dict(task,folder=f'attempts/{group}/a{attempt}')
            folder=OUT/task['folder'];folder.mkdir(parents=True,exist_ok=True);write(folder/'task.json',task)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_USE_V1='1',
                VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            proc=None;error=None;started=time.time()
            try:
                with (folder/'worker.log').open('a') as log:
                    proc=subprocess.Popen(['taskset','-c',','.join(map(str,m['affinity'][str(g)])),sys.executable,'-B','-u',str(HERE/'run.py'),
                        '--worker',str(folder/'task.json'),'--gpu',str(g)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while proc.poll() is None:
                        if time.time()>=deadline:error='deadline';break
                        if time.time()-started>len(task['jobs'])*d['episode_timeout_s']+1800:error='group_watchdog';break
                        external=infra.compute_pids(m['hardware'][str(g)]['uuid'])-{proc.pid}
                        if external:
                            error='external_gpu_process';write(folder/'contamination.json',dict(external=sorted(external),updated=time.time()));break
                        time.sleep(2)
                    if error is None and proc.returncode!=0:error=f'worker_exit_{proc.returncode}'
            except Exception:error=traceback.format_exc()
            finally:
                if proc is not None and proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            complete=(folder/'worker_complete.json').exists() and error is None
            result=dict(group=group,gpu=g,attempt=attempt,folder=task['folder'],cell=task['cell'],state='complete' if complete else 'failed',error=error,started=started,ended=time.time())
            write(folder/'attempt_status.json',result)
            with mutex:
                active.pop(str(g),None)
                if complete:
                    counts['complete']+=1;finished.add(group);write(OUT/f'groups/{group}.json',result)
                else:
                    cal=OUT/f"calibration/g{g}_c{task['cell']['capacity']}/status.json"
                    if (cal.exists() and not read(cal)['passed']) or (cal.parent/'runs').exists() and not cal.exists():disabled.add(f"{g}:{task['cell']['capacity']}")
                    # Whole group recovery for interference/calibration only, never an unfavorable metric.
                    recover=error=='external_gpu_process' or f"{g}:{task['cell']['capacity']}" in disabled
                    if recover and attempt<d['max_group_retries'] and time.time()<deadline:
                        pending.append(dict(task,attempt=attempt+1));counts['retried']+=1
                    else:counts['failed']+=1;write(OUT/f'groups/{group}.json',result)
                status()
            write(OUT/f'worker_g{g}.json',dict(state='idle',updated=time.time()))
    try:
        with ThreadPoolExecutor(max_workers=len(gs)) as pool:
            for future in [pool.submit(consume,g) for g in gs]:future.result()
    finally:
        if sim.poll() is None:
            remaining=max(0,min(30,deadline-time.time()))
            try:sim.wait(timeout=remaining)
            except subprocess.TimeoutExpired:sim.terminate();sim.wait(timeout=15)
        simlog.close()
        write(OUT/'simulator_process.json',dict(returncode=sim.returncode,complete=(OUT/'simulator/full_complete.json').exists()))
    final='complete' if counts['complete']==len(m['tasks']) else 'deadline_reached' if time.time()>=deadline else 'finished_with_failures'
    write(OUT/'status.json',dict(state=final,counts=counts,planned_groups=len(m['tasks']),unfinished_groups=[t['group'] for t in pending],updated=time.time(),deadline=deadline))
    lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.worker:worker(a.worker,a.gpu)
    else:
        try:controller()
        except Exception:write(OUT/'status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()));raise
