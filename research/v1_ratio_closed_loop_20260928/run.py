"""GPU full-loop evaluation with existing bounded idle-only controller."""
import argparse,gzip,os,random,statistics as st,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_ratio_closed_loop_20260928.common import *
from research.v1_ratio_closed_loop_20260928.audit import audit
from research.v1_fixed_order_20260927.episode import episode
from research.v1_candidate_audit_20260928 import run as infrastructure
infrastructure.OUT=OUT;infrastructure.HERE=HERE;infrastructure.verify=verify

def save(engine,s,reqs,cfg,folder,name,meta):
    p=folder/'runs'/f'{name}.json';assert not p.exists()
    write(folder/'current.json',dict(name=name,state='running',updated=time.time()))
    r=episode(engine,s,reqs,'threshold',cfg,timeout=180);rows=r.pop('rounds');r.update(config=cfg,**meta)
    rp=folder/'rounds'/f'{name}.json.gz';rp.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(rp,'wt',compresslevel=1) as f:json.dump(rows,f,separators=(',',':'))
    r.update(rounds_path=str(rp.relative_to(OUT)),rounds_sha256=digest(rp));write(p,r)
    assert r['status']=='complete',r['error']
    r['audit']=audit(r,rows);write(p,r)
    write(folder/'current.json',dict(name=name,state='saved',updated=time.time()))
    print(name,round(r['metrics']['mean_ttft_ms'],3),r['audit']['different_from_fixed64'],flush=True)
    return r

def worker(key,gpu):
    from research.v1_state_aware_20260923 import run as infra
    lock=infra.acquire(f'/tmp/query-capacity-gpu{gpu}.lock');assert infra.idle(infra.hardware()[gpu])
    d=verify();g=next(g for g in d['groups'] if g['key']==key);folder=OUT/'groups'/key
    manifest=read(OUT/'manifest.json');assert not (folder/'status.json').exists()
    status=dict(group=g,gpu=gpu,pid=os.getpid(),state='running',updated=time.time());write(folder/'status.json',status)
    os.sched_setaffinity(0,manifest['affinity'][str(gpu)])
    try:
        import torch
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        assert str(torch.cuda.get_device_properties(0).uuid).lower()==manifest['hardware'][str(gpu)]['uuid'].removeprefix('GPU-').lower()
        engine=LLMEngine.from_engine_args(EngineArgs(model=MODEL,dtype='half',max_model_len=8176,
            max_num_seqs=64,max_num_batched_tokens=2048,enable_chunked_prefill=True,enable_prefix_caching=False,
            block_size=16,gpu_memory_utilization=.9,enforce_eager=True,disable_log_stats=True,seed=0,
            num_gpu_blocks_override=g['capacity'],scheduler_cls='research.v1_ratio_closed_loop_20260928.scheduler.RatioScheduler'))
        s=engine.engine_core.engine_core.scheduler
        cfg=dict(budget=g['budget'],chunk=g['chunk'],concurrency=32,threshold=spec('fixed64'))
        burst=read(OUT/f"inputs/{g['input']}_burst_stop.json")
        warm=episode(engine,s,burst,'threshold',cfg,timeout=180);assert warm['status']=='complete'
        aa=[save(engine,s,burst,cfg,folder,f'aa_{r}',dict(phase='aa',arm='fixed64',shape='burst_stop',repeat=r)) for r in range(2)]
        vals=[r['metrics']['mean_ttft_ms'] for r in aa];spread=(max(vals)-min(vals))/st.fmean(vals)
        write(folder/'aa.json',dict(relative_spread=spread,values=vals,passed=spread<=.05));assert spread<=.05,'A/A failed'
        for shape in SHAPES:
            reqs=read(OUT/f"inputs/{g['input']}_{shape}.json")
            warm=episode(engine,s,reqs,'threshold',cfg,timeout=180);assert warm['status']=='complete'
            order=list(ARMS);random.Random(20260928+sum(map(ord,key+shape))).shuffle(order)
            for repeat in range(2):
                for arm in (order if repeat==0 else list(reversed(order))):
                    save(engine,s,reqs,dict(cfg,threshold=spec(arm)),folder,f'{shape}_{arm}_r{repeat}',
                         dict(phase='formal',arm=arm,shape=shape,repeat=repeat))
        write(folder/'status.json',dict(status,state='complete',formal_runs=12,updated=time.time()))
    except BaseException:
        write(folder/'status.json',dict(status,state='failed',error=traceback.format_exc(),updated=time.time()));raise
    finally:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.worker:worker(a.worker,a.gpu)
    else:infrastructure.controller()
