"""Replayed synthetic prefixes; one timed action followed by verified abort/reset."""
import random
import statistics as st
import time
from research.v1_efficient_points_20260927.common import *

GRID=[32,64,128,192,224,240,248,255,256,257,264,272,288,320,448,480,496,511,512,513,528,576,768,1024]
CONTEXTS={'fresh':0,'long':8,'mixed':6}
CHUNKS=[256,64]
REPEATS=4


def requests(context):
    shorts=[dict(id=f'd{i}',prompt=256,output=64,arrival_ms=0) for i in range(8)] if context=='mixed' else []
    return shorts+[dict(id=f'p{i}',prompt=6144,output=16,arrival_ms=0) for i in range(32-len(shorts))]


def probe(engine,s,context,q,k):
    from vllm import SamplingParams
    s.arrival_clock=lambda:0.
    s.configure('threshold',**BASE,threshold=dict(spec('current'),profile_action=dict(budget=1024,chunk=256)))
    reqs=requests(context)
    try:
        for r in reqs:
            engine.add_request(r['id'],{'prompt_token_ids':[100]*r['prompt']},
                SamplingParams(temperature=0.,max_tokens=r['output'],ignore_eos=True,detokenize=False),arrival_time=0.)
        for i in range(CONTEXTS[context]+1):
            if i==CONTEXTS[context]:
                s.threshold_spec=dict(spec('current'),profile_action=dict(budget=q,chunk=k))
            start=time.perf_counter();engine.step();elapsed=(time.perf_counter()-start)*1000
            s.round_rows[-1]['step_ms']=elapsed
        row=s.round_rows[-1]
        assert sum(row['scheduled'].values())==q,(context,q,k,row['scheduled'])
        assert row['verified'] and row['preemptions']==0
        fingerprint=row['state_fingerprint']
        active=[r for r in row['state'] if r['rid'] in row['scheduled']]
        avg_context=sum(r['computed']*row['scheduled'][r['rid']] for r in active)/q
        return dict(context=context,q=q,chunk=k,step_ms=elapsed,schedule_ms=row['schedule_ms'],
                    non_schedule_ms=elapsed-row['schedule_ms'],avg_context=avg_context,
                    scheduled_requests=len(active),decode_requests=sum(r['outputs']>0 for r in active),
                    fingerprint=fingerprint,row=row,
                    prefix=[dict(fingerprint=r['state_fingerprint'],scheduled=r['scheduled']) for r in s.round_rows[:-1]])
    finally:
        engine.abort_request([r['id'] for r in reqs])
        assert not engine.has_unfinished_requests() and not s.requests and not s.running and not s.waiting
        assert s.kv_cache_manager.block_pool.get_num_free_blocks()==s.kv_cache_config.num_blocks-1


def run_profile(engine,s,folder):
    records=[]
    for context in CONTEXTS:
        order=[(q,k) for q in GRID for k in CHUNKS]
        random.Random(20260927+sum(map(ord,context))).shuffle(order)
        # One excluded warm-up of every shape/budget, then 4 balanced passes.
        fingerprints=set();prefix=None
        for repeat in range(-1,REPEATS):
            for q,k in (order if repeat%2==0 else list(reversed(order))):
                result=probe(engine,s,context,q,k)
                fingerprints.add(result['fingerprint'])
                assert len(fingerprints)==1,'Same-state profiling mismatch'
                if prefix is None:prefix=result['prefix']
                assert result['prefix']==prefix,'Prefix replay mismatch'
                result.update(repeat=repeat,warmup=repeat==-1)
                records.append(result)
                # Each probe is independently recoverable on interruption.
                write(folder/f'profile/{context}_q{q}_k{k}_r{repeat}.json',result)
                write(folder/'current.json',dict(state='profiling',context=context,q=q,chunk=k,repeat=repeat,
                    completed=len(records),expected=len(CONTEXTS)*len(GRID)*len(CHUNKS)*(REPEATS+1),updated=time.time()))
            print('profile',context,'repeat',repeat,'saved',len(records),flush=True)
    # Timing data are frozen before any formal TTFT evaluation.
    write(folder/'profile_summary.json',dict(probes=len(records),measured=sum(not r['warmup'] for r in records),
        matched_prefix=True,reset_checked=True,grid=GRID))
    s.arrival_clock=time.perf_counter


def freeze_calibration(folders):
    assert not (OUT/'calibration.json').exists()
    all_rows=[]
    for folder in folders:
        rr=[read(p) for p in sorted((folder/'profile').glob('*.json'))]
        assert len(rr)==len(CONTEXTS)*len(GRID)*len(CHUNKS)*(REPEATS+1)
        for r in rr:
            if not r['warmup']:all_rows.append(dict(r,group=folder.name))
    curves={};cells=[];contexts={}
    for ctx in CONTEXTS:
        contexts[ctx]=st.median(r['avg_context'] for r in all_rows if r['context']==ctx)
        for k in CHUNKS:
            curve=[]
            for q in GRID:
                medians=[]
                for folder in folders:
                    rr=[r for r in all_rows if r['group']==folder.name and r['context']==ctx and r['chunk']==k and r['q']==q]
                    assert len(rr)==REPEATS
                    medians.append(st.median(r['non_schedule_ms'] for r in rr))
                ms=st.median(medians)
                curve.append(dict(q=q,ms=ms,card_medians=medians))
                cells.append(dict(context=ctx,chunk=k,q=q,ms=ms))
            curves[f'{ctx}_k{k}']=curve
    import math
    selected={};scores=[]
    for k in CHUNKS:
        selected[k]=[]
        for low,high in [(192,320),(448,576)]:
            candidates=[]
            for q in GRID:
                if not low<=q<=high:continue
                # Geometric mean throughput across contexts. No TTFT outcome
                # enters selection. This is only a candidate generator.
                score=math.exp(st.fmean(math.log(q/next(r['ms'] for r in curves[f'{ctx}_k{k}'] if r['q']==q)) for ctx in CONTEXTS))
                candidates.append((score,q));scores.append(dict(chunk=k,low=low,high=high,q=q,score=score))
            selected[k].append(max(candidates)[1])
    a,b=selected[256];c,d=selected[64]
    # Keep original actions as explicit fallbacks; deduplicate exact pairs.
    pairs=list(dict.fromkeys([(a,256),(b,256),(1024,64),(d,64),(c,64),
                             (256,256),(512,256),(512,64)]))
    result=dict(frozen_at=time.time(),profile_groups=[p.name for p in folders],
        profile_summary_hashes={p.name:digest(p/'profile_summary.json') for p in folders},
        pairs=pairs,selected=selected,curves=curves,contexts=contexts,scores=scores,
        rule='geometric-mean measured throughput per predefined neighborhood; no formal TTFT used',
        timing='step minus scheduler; not isolated CUDA kernel timing')
    write(OUT/'calibration.json',result)
    print('Frozen measured candidate pairs:',pairs,flush=True)
    return result
