"""Native-allocator discrete-event simulator; timings explicitly uncalibrated."""
import argparse,hashlib,importlib.util,json,math,statistics as st,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_ratio_closed_loop_20260928.common import *
from research.v1_ratio_closed_loop_20260928.policy import select,pairs
from research.v1_budget_20260922.policy import State
from research.v1_broad12h_20260925.policy import select as legacy

def timing_model():
    # Load only the repository timing module; the native allocator replaces the
    # separate simulator engine so CPU and GPU physical semantics stay identical.
    sp=importlib.util.spec_from_file_location('ratio_sim_timing',ROOT/'simulator_core/timing.py')
    mod=importlib.util.module_from_spec(sp);sys.modules[sp.name]=mod;sp.loader.exec_module(mod)
    model=mod.ResourceTiming(dict(measurement_scope='synthetic-uncalibrated',parameters=dict(
        launch_ms=1.,weight_read_ms=5.,prefill_ms_per_token=.08,prefill_context_ms_per_token2=.000003,
        decode_ms_per_request=.03,decode_context_ms_per_token=.00001)))
    return model,mod.TimingBatch

def simulate(reqs,capacity,budget,chunk,arm,profile):
    from vllm import SamplingParams
    from vllm.v1.request import Request,RequestStatus
    from research.v1_budget_20260922.test_scheduler import scheduler
    from research.v1_ratio_closed_loop_20260928.scheduler import RatioScheduler
    model,Batch=timing_model();s=scheduler(blocks=capacity,cls=RatioScheduler);s.max_model_len=8176
    clock=[0.];s.arrival_clock=lambda:clock[0]/1000
    s.configure('threshold',budget=budget,chunk=chunk,concurrency=32,threshold=spec(arm))
    targets={r['id']:r['output'] for r in reqs};first={};cursor=0;wall=time.monotonic();checks=0
    digest0=hashlib.sha256();counts={str(i):0 for i in range(-1,4)}
    preemptions=recomputed=executed=changed=0
    while cursor<len(reqs) or s.requests:
        assert checks<20000 and time.monotonic()-wall<180,'Simulation limit'
        if not s.requests and cursor<len(reqs):clock[0]=max(clock[0],reqs[cursor]['arrival_ms'])
        while cursor<len(reqs) and reqs[cursor]['arrival_ms']<=clock[0]:
            r=reqs[cursor];s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,
                SamplingParams(max_tokens=r['output']),None,None,arrival_time=r['arrival_ms']/1000));cursor+=1
        out=s.schedule();row=s.round_rows[-1];states=[State(**r) for r in row['state']]
        assert row['free_before']+sum(r.blocks for r in states)==capacity-1
        args=(states,budget,chunk,row['free_before'],capacity,16,32,8176)
        ref=legacy(*args,threshold=spec('fixed64'),recent=dict(arrival_rate=row['features']['arrival_rate']))
        own=select(*args,threshold=spec(arm),recent=dict(arrival_rate=row['features']['arrival_rate']))
        assert own['threshold']==ref['threshold']==row['threshold']
        if arm=='fixed64' or pairs(budget,chunk,arm)==pairs(budget,chunk,'fixed64'):
            assert (own['budget'],own['chunk'])==(ref['budget'],ref['chunk'])
            assert (None if own['selected'] is None else own['selected'].tokens)==(None if ref['selected'] is None else ref['selected'].tokens)
        if ref['selected'] is not None:changed+=row['scheduled']!=ref['selected'].tokens
        if row['intervened']:assert row['baseline_growth_blocks']-row['actual_growth_blocks']>=row['threshold']
        pre=[];ctx=[];dec=[]
        for r in states:
            n=row['scheduled'].get(r.rid,0)
            if not n:continue
            # A preempted request can have a stale pre-step computed count;
            # actual allocation.before is authoritative for executed attention.
            before=next(a['before'] for a in row['allocations'] if a['id']==r.rid)
            if r.decode and n==1:dec.append(before)
            else:pre.append(n);ctx.append(before)
        duration,_=model.estimate(Batch(tuple(pre),tuple(ctx),tuple(dec)))
        if profile=='resource_staircase':duration+=5*math.ceil(sum(row['scheduled'].values())/256)
        clock[0]+=max(duration,.001);done=[]
        for rid in out.num_scheduled_tokens:
            r=s.requests[rid]
            if r.num_computed_tokens==r.num_tokens:
                r.append_output_token_ids(100);first.setdefault(rid,clock[0])
            if len(r.output_token_ids)==targets[rid]:done.append(rid)
        s.finish_requests(done,RequestStatus.FINISHED_LENGTH_CAPPED)
        preemptions+=row['preemptions'];recomputed+=row['recomputed_tokens'];executed+=sum(row['scheduled'].values())
        counts[str(row['features']['selected_candidate'])]+=1;checks+=1
        digest0.update(json.dumps([row['scheduled'],row['free_before'],row['free_after'],clock[0]],sort_keys=True).encode())
        # The scheduler uses only the last eight rounds for recent features;
        # no branches or round-index-dependent policies exist in this simulator.
        if len(s.round_rows)>16:s.round_rows=s.round_rows[-8:]
    assert len(first)==len(reqs) and not s.requests
    assert s.kv_cache_manager.block_pool.get_num_free_blocks()==capacity-1
    assert executed-sum(r['prompt']+r['output']-1 for r in reqs)==recomputed
    return dict(complete=True,mean_ttft_ms=st.fmean(first[r['id']]-r['arrival_ms'] for r in reqs),
        rounds=checks,preemptions=preemptions,recomputed_tokens=recomputed,executed_tokens=executed,
        selected_candidates=counts,different_from_fixed64=changed,trajectory_sha256=digest0.hexdigest(),
        elapsed_simulated_ms=clock[0],cpu_wall_s=time.monotonic()-wall,profile=profile,
        timing_claim='synthetic-uncalibrated; not measured GPU performance')

def smoke():
    verify();results=[]
    reqs=[dict(id=f'r{i:03}',prompt=(128,800,1800,2500)[i%4]+i,output=24,arrival_ms=0.) for i in range(20)]
    for cap in (192,768,1536):
        for budget,chunk in CONFIGS:
            for arm in ARMS:
                r=simulate(reqs,cap,budget,chunk,arm,'resource');results.append(dict(capacity=cap,budget=budget,chunk=chunk,arm=arm,**r))
    write(OUT/'validation.json',dict(passed=True,source_hashes=sources(),checks=results,
        note='Native CPU physical semantics; artificial clock; fixed64 equivalent to frozen legacy'))
    print('Native smoke passed',len(results),'trajectories',sum(r['rounds'] for r in results),'rounds',flush=True)

def full():
    d=verify();records=[];started=time.time();cohorts={g['input']:(g['source'],g['window'],g['capacity']) for g in d['groups']}
    for name,(source,window,cap) in cohorts.items():
        for shape in SHAPES:
            reqs=read(OUT/f'inputs/{name}_{shape}.json')
            for budget,chunk in SIM_CONFIGS:
                for profile in ('resource','resource_staircase'):
                    controls={}
                    for arm in ARMS:
                        key=f'{name}_{shape}_t{budget}_k{chunk}_{profile}_{arm}'
                        path=OUT/f'simulation/{key}.json'
                        if path.exists():
                            r=read(path);assert r['source_hashes']==d['source_hashes']
                        else:
                            r=dict(source=source,window=window,shape=shape,capacity=cap,budget=budget,chunk=chunk,arm=arm,
                                source_hashes=d['source_hashes'],**simulate(reqs,cap,budget,chunk,arm,profile));write(path,r)
                        controls[arm]=r;records.append({k:v for k,v in r.items() if k!='source_hashes'})
                    for arm in ARMS[1:]:
                        if pairs(budget,chunk,arm)==pairs(budget,chunk,'fixed64'):
                            assert controls[arm]['trajectory_sha256']==controls['fixed64']['trajectory_sha256']
                            assert controls[arm]['mean_ttft_ms']==controls['fixed64']['mean_ttft_ms']
                    write(OUT/'simulation_status.json',dict(saved=len(records),expected=528,updated=time.time()))
            print('SIM',name,shape,len(records),'/528',flush=True)
    assert len(records)==528 and all(r['complete'] for r in records)
    write(OUT/'simulation_summary.json',dict(complete=True,runs=len(records),elapsed_s=time.time()-started,rows=records,
        source_hashes=d['source_hashes'],performance_calibrated=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    smoke() if a.smoke else full()
