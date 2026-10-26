"""Bounded same-state GPU diagnostics; never optimize coefficients on results."""
import argparse
import gzip
import os
from pathlib import Path
import statistics as st
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_mechanism_models_replicas_20260925.common import *


def episode(engine,scheduler,reqs,config,timeout=300):
    from vllm import SamplingParams
    scheduler.configure('threshold',**config)
    start=time.perf_counter()
    clocks={r['id']:dict(first_ms=None,finish_ms=None,tokens=0) for r in reqs}
    for r in reqs:
        engine.add_request(r['id'],{'prompt_token_ids':[100]*r['prompt']},
            SamplingParams(temperature=0.,max_tokens=r['output'],ignore_eos=True,detokenize=False))
    while engine.has_unfinished_requests():
        assert time.perf_counter()-start<timeout,'Probe episode deadline'
        before=time.perf_counter();outputs=engine.step();now=(time.perf_counter()-start)*1000
        scheduler.round_rows[-1].update(step_ms=(time.perf_counter()-before)*1000,step_end_ms=now)
        for output in outputs:
            c=clocks[output.request_id];n=len(output.outputs[0].token_ids) if output.outputs else 0
            if n>c['tokens']:
                assert n==c['tokens']+1;c['tokens']=n
                if c['first_ms'] is None:c['first_ms']=now
            if output.finished:c['finish_ms']=now
    assert all(clocks[r['id']]['tokens']==r['output'] and clocks[r['id']]['finish_ms'] is not None for r in reqs)
    result=dict(clocks=clocks,mean_ttft_ms=st.fmean(c['first_ms'] for c in clocks.values()),
        preemptions=sum(r['preemptions'] for r in scheduler.round_rows),
        recomputed_tokens=sum(r['recomputed_tokens'] for r in scheduler.round_rows),
        interventions=sum(r['intervened'] for r in scheduler.round_rows),
        rounds=scheduler.round_rows)
    if scheduler.probe_target is not None:
        row=scheduler.round_rows[scheduler.probe_target]
        branch_ms=(row['probe_decision_wall']-start)*1000
        pending=[r['rid'] for r in row['state'] if r['outputs']==0]
        assert pending and all(clocks[rid]['first_ms']>=branch_ms for rid in pending)
        result.update(branch_ms=branch_ms,pending_ids=pending,
                      residual_waiting_area_ms=sum(clocks[rid]['first_ms']-branch_ms for rid in pending),
                      branch_fingerprint=row['state_fingerprint'],branch_scheduled=row['scheduled'],
                      branch_free_after=row['free_after'],branch_applied=row['branch_applied'])
    return result


def save(folder,name,result):
    rounds=result.pop('rounds');path=folder/'rounds'/f'{name}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',compresslevel=1) as stream:json.dump(rounds,stream,separators=(',',':'))
    result.update(rounds_path=str(path.relative_to(OUT)),rounds_sha256=digest(path))
    write(folder/f'{name}.json',result)
    return rounds


def worker(model,capacity,gpu):
    from research.v1_state_aware_20260923 import run as infra
    lock=infra.acquire(f'/tmp/query-capacity-gpu{gpu}.lock')
    assert infra.idle(infra.hardware()[gpu])
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_USE_V1='1',
                      VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    manifest=read(OUT/'manifest.json');design=read(OUT/'design.json')
    os.sched_setaffinity(0,manifest['affinity'][str(gpu)])
    plan=read(OUT/'probes/design.json')
    for relative,sha in plan['source_hashes'].items():assert digest(ROOT/relative)==sha
    folder=OUT/'probes'/f'{model}_c{capacity}'
    assert not (folder/'status.json').exists(),'No overwrite of probe results'
    write(folder/'status.json',dict(state='running',model=model,capacity=capacity,gpu=gpu,updated=time.time()))
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        engine=LLMEngine.from_engine_args(EngineArgs(model=design['models'][model]['path'],dtype='half',
            max_model_len=8176,max_num_seqs=64,max_num_batched_tokens=2048,enable_chunked_prefill=True,
            enable_prefix_caching=False,block_size=16,gpu_memory_utilization=.9,enforce_eager=True,
            disable_log_stats=True,seed=0,num_gpu_blocks_override=capacity,
            scheduler_cls='research.v1_mechanism_models_replicas_20260925.probe_scheduler.ProbeScheduler'))
        s=engine.engine_core.engine_core.scheduler
        reqs=read(OUT/'inputs/conversation_w0_burst_stop.json')
        assert all(r['arrival_ms']==0 for r in reqs)
        base=dict(BASE,threshold=formula(1))
        warm=[dict(id=f'pw{i}',prompt=256,output=16,arrival_ms=0) for i in range(4)]
        for _ in range(2):episode(engine,s,warm,base)
        # Full unmeasured warmup avoids using a first-shape run as its own comparison.
        episode(engine,s,reqs,base)
        anchor=episode(engine,s,reqs,base);anchor_rounds=save(folder,'anchor',anchor)
        candidates=[r for r in anchor_rounds if r['intervened'] and any(x['outputs']==0 for x in r['state'])]
        candidates.sort(key=lambda r:(r['features']['pressure'],r['round']))
        selected=[]
        for fraction in (.25,.75):
            if not candidates:break
            target=candidates[int(fraction*(len(candidates)-1))]
            if target['round'] not in [r['round'] for r in selected]:selected.append(target)
        times=[r['features']['arrival_time_s'] for r in anchor_rounds]
        write(folder/'selected_states.json',dict(frozen_before_branch_outcomes=True,selection='25th/75th pressure rank of actual saving interventions',
            available=len(candidates),states=[dict(round=r['round'],fingerprint=r['state_fingerprint'],features=r['features'],
                saving=r['baseline_growth_blocks']-r['actual_growth_blocks'],scheduled=r['scheduled'],baseline=r['baseline']) for r in selected]))
        summaries=[]
        for row in selected:
            index=row['round'];results={}
            for repeat in (0,1):
                for arm in (('arrival','off') if repeat==0 else ('off','arrival')):
                    s.probe_times_context=times;s.probe_target_context=index
                    spec=formula(1) if arm=='arrival' else 'off'
                    config=dict(BASE,threshold=formula(1),branch=dict(round=index,fingerprint=row['state_fingerprint'],threshold=spec))
                    result=episode(engine,s,reqs,config)
                    assert result['branch_applied'] and result['branch_fingerprint']==row['state_fingerprint']
                    assert result['branch_scheduled']==(row['scheduled'] if arm=='arrival' else row['baseline'])
                    for j in range(index+1):assert result['rounds'][j]['state_fingerprint']==anchor_rounds[j]['state_fingerprint']
                    result.update(arm=arm,repeat=repeat,target_round=index,gpu=gpu,model=model,capacity=capacity)
                    save(folder,f'branch_r{index}_{arm}_rep{repeat}',result);results[arm,repeat]=result
            area={arm:st.fmean(results[arm,r]['residual_waiting_area_ms'] for r in (0,1)) for arm in ('arrival','off')}
            summaries.append(dict(round=index,fingerprint=row['state_fingerprint'],pending=len(results['arrival',0]['pending_ids']),
                saving_blocks=row['baseline_growth_blocks']-row['actual_growth_blocks'],
                arrival_residual_area_ms=area['arrival'],off_residual_area_ms=area['off'],
                saved_waiting_area_ms=area['off']-area['arrival'],
                equivalent_mean_ttft_saved_ms=(area['off']-area['arrival'])/len(reqs),
                gain_r0_area_ms=results['off',0]['residual_waiting_area_ms']-results['arrival',0]['residual_waiting_area_ms'],
                gain_r1_area_ms=results['off',1]['residual_waiting_area_ms']-results['arrival',1]['residual_waiting_area_ms']))
        write(folder/'summary.json',dict(model=model,capacity=capacity,states=summaries,
            claim='Single-action causal probe conditional on matched physical/policy state and replayed prefix estimator time; all-arrived workload. Not a stationary/live-arrival performance benchmark.'))
        write(folder/'status.json',dict(state='complete',states=len(selected),model=model,capacity=capacity,gpu=gpu,updated=time.time()))
    except BaseException:
        import traceback
        write(folder/'status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()))
        raise
    finally:lock.close()


def freeze():
    assert not (OUT/'probes/design.json').exists()
    paths=[HERE/'probes.py',HERE/'probe_scheduler.py']
    plan=dict(created=time.time(),tasks=[dict(model=m,capacity=c) for c in (768,4096) for m in ('qwen05b','smol360m','qwen7b')],
        repeats=2,max_selected_states=2,global_deadline='Same deadline as main pilot; no extension',
        workload='conversation_w0_burst_stop, all arrivals at zero',
        selection='25th/75th pressure rank among actual arrival-policy interventions; no outcome-based selection',
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in paths},main_sources=sources())
    write(OUT/'probes/design.json',plan)
    print('Frozen six model/capacity probe groups, at most 48 paired-branch runs plus six anchors and warmups',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--model')
    parser.add_argument('--capacity',type=int);parser.add_argument('--gpu',type=int);args=parser.parse_args()
    if args.freeze:freeze()
    else:worker(args.model,args.capacity,args.gpu)
