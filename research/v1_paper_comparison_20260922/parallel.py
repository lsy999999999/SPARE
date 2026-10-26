"""Three-GPU execution with physical GPU fixed within each comparison block."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.v1_paper_comparison_20260922 import run as core

OUT = ROOT/'artifacts/v1_paper_comparison_multi_20260922_r3'
core.OUT = OUT
core.ref.OUT = OUT
GPUS = (2,6,7)
AFFINITY = {2:'9,13,15',6:'28,31,36',7:'40,41,52'}
TUNE_GPU = {'queue':2,'burst':6,'restricted':7}
# Sorted blocks: code0, code1, conversation0, conversation1.
COMPARE_GPU = {'queue':(6,7,6,7),'burst':(2,6,7,6),'restricted':(2,2,6,7)}
WORKER_GPU = None
BASE_JOBS = core.jobs
read,write,digest = core.read,core.write,core.digest


def jobs(m,phase,regime,selection=None):
    base = BASE_JOBS(m,phase,regime,selection)
    result = []
    blocks = sorted(k for k,b in m['blocks'].items() if b['role']=='compare')
    for job in base:
        if phase in ('smoke','aa'):
            targets = GPUS
        elif phase=='tune':
            targets = (TUNE_GPU[regime],)
        else:
            targets = (COMPARE_GPU[regime][blocks.index(job['block'])],)
        for gpu in targets:
            if WORKER_GPU is None or gpu==WORKER_GPU:
                result.append(dict(job,key=job['key']+f'_g{gpu}',gpu=gpu,
                    gpu_uuid=m['hardware_by_gpu'][str(gpu)]['uuid']))
    return result


def initialize():
    old = read(core.OLD/'manifest.json')
    assert read(core.OLD/'summary.json')['complete']
    assert str(core.ref.MODEL)==old['model']
    for name in ('policy.py','scheduler.py','run.py'):
        assert digest(ROOT/'research/v1_budget_20260922'/name)==old['source_hashes']['research/v1_budget_20260922/'+name]
    for p,v in old['versions'].items():
        assert importlib.metadata.version(p)==v
    dist = importlib.metadata.distribution('vllm')
    for name,sha in old['source_hashes'].items():
        if name.startswith('vllm/'):
            assert digest(dist.locate_file(name))==sha
    hardware = {}
    for gpu in GPUS:
        os.environ['V1_BUDGET_GPU']=str(gpu)
        hardware[str(gpu)] = core.ref.hardware()
        assert hardware[str(gpu)]['name']=='NVIDIA GeForce RTX 4090'
    os.environ['V1_BUDGET_GPU']='2'
    assert hardware['2']==old['hardware']
    m = dict(schema=1,implementation='a-paper-rules-v1/three-gpu-r3',
        hardware=dict(mode='GPU fixed within comparison block',gpus=list(hardware.values())),
        hardware_by_gpu=hardware,cpu_affinity_by_gpu={str(k):v for k,v in AFFINITY.items()},
        model=old['model'],model_revision=old['model_revision'],versions=old['versions'],
        blocks=old['blocks'],histories=core.history_data(old),
        reference=str(core.OLD),
        reference_hashes={n:digest(core.OLD/n) for n in ('manifest.json','summary.json','selection.json','capacity.json','tuning_report.json')},
        archived_tuning_hashes={p.name:digest(p) for p in sorted((core.OLD/'runs').glob('tune_*.json'))},
        inherited_selections=read(core.OLD/'selection.json'),paper_configs=core.CONFIGS,
        arms=list(core.ARMS),regimes=list(core.REGIMES),repeats=2,timeout_s=300,
        capacity=read(core.OLD/'capacity.json')['blocks'],
        tuning_gpu=TUNE_GPU,comparison_gpu={k:list(v) for k,v in COMPARE_GPU.items()},
        source_hashes={p.name:digest(p) for p in sorted(core.HERE.iterdir()) if p.is_file()},
        frozen_reference_hashes=old['source_hashes'],
        expected=dict(smoke=6,aa=54,tune=36,compare=96),
        claim='Exploratory reused source windows; all four methods paired within the same physical GPU per block')
    path=OUT/'manifest.json'
    if path.exists():
        assert read(path)==m,'Frozen manifest changed'
    else:
        write(path,m)
        for p in core.HERE.iterdir():
            if p.is_file():
                dest=OUT/'source'/p.name
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(p.read_bytes())
        write(OUT/'capacity.json',dict(blocks=m['capacity']))
    return m


def worker(m,phase,regime,gpu):
    global WORKER_GPU
    WORKER_GPU=gpu
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    assert os.environ['V1_BUDGET_GPU']==str(gpu)
    assert sorted(os.sched_getaffinity(0))==list(map(int,AFFINITY[gpu].split(',')))
    assert core.ref.hardware()==m['hardware_by_gpu'][str(gpu)]
    for name,sha in m['source_hashes'].items():
        assert digest(core.HERE/name)==sha
    # Engine construction verifies CUDA UUID independently.
    local=dict(m,hardware=m['hardware_by_gpu'][str(gpu)])
    core.phase_run(local,phase,regime)


def run_gpu(m,phase,gpu):
    selection=read(OUT/'selection.json') if phase=='compare' else None
    for regime in (('restricted',) if phase=='smoke' else core.REGIMES):
        planned=[j for j in jobs(m,phase,regime,selection) if j['gpu']==gpu]
        attempts=0
        while any(not (OUT/'runs'/(j['key']+'.json')).exists() for j in planned):
            used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used>1024:
                raise RuntimeError(f'GPU {gpu} occupied: {used} MiB')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),V1_BUDGET_GPU=str(gpu))
            log=OUT/'logs'/f'{phase}_{regime}_g{gpu}.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as f:
                code=subprocess.run(['taskset','-c',AFFINITY[gpu],sys.executable,'-B','-u',
                    str(core.HERE/'parallel.py'),'--phase',phase,'--regime',regime,
                    '--gpu',str(gpu)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=21600).returncode
            if code not in (0,75):
                raise RuntimeError(f'{phase}/{regime}/GPU{gpu}: exit {code}; see {log}')
            attempts+=1
            if attempts>len(planned)+1:
                raise RuntimeError('No forward progress')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('all','smoke','aa','tune','compare','summary'),default='all')
    p.add_argument('--regime',choices=core.REGIMES,default='restricted')
    p.add_argument('--gpu',type=int,choices=GPUS)
    args=p.parse_args()
    core.jobs=jobs
    if args.gpu is not None:
        worker(read(OUT/'manifest.json'),args.phase,args.regime,args.gpu)
        return
    m=initialize()
    if args.phase=='summary':
        core.summarize(m)
        return
    with (OUT/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for phase in ('smoke','aa','tune','compare'):
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures=[pool.submit(run_gpu,m,phase,g) for g in GPUS]
                for f in futures:
                    f.result()
            if phase=='aa':
                core.check_aa(m)
            elif phase=='tune':
                core.select(m)
        core.summarize(m)


if __name__=='__main__':
    main()
