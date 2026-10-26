"""Differential tests and full native KV-allocator simulation, not latency predictions."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_arrival_kv_papers_20260925.common import *
from research.v1_arrival_kv_papers_20260925.scheduler import MatrixScheduler
from research.v1_broad12h_20260925.scheduler import FormulaScheduler
from research.v1_paper_comparison_20260922.scheduler import ComparisonScheduler,PAPER_METHODS
from research.v1_budget_20260922.scheduler import BudgetScheduler
from research.v1_budget_20260922.test_scheduler import scheduler,add,advance
from vllm import SamplingParams
from vllm.v1.request import Request,RequestStatus

def equivalence():
 n=0
 for arm,spec in ARMS.items():
  cls=FormulaScheduler if spec['method']=='threshold' else ComparisonScheduler if spec['method'] in PAPER_METHODS else BudgetScheduler
  for cap in (48,96):
   a,b=[scheduler(blocks=cap,cls=c) for c in (MatrixScheduler,cls)]
   for s in (a,b):
    s.history_context=[24]*1000;s.seed_context=42;s.arrival_clock=lambda:0.
    s.configure(spec['method'],**spec['config']);add(s,[257,129,385,99,400,12])
   for step in range(4000):
    x,y=a.schedule(),b.schedule()
    assert x.num_scheduled_tokens==y.num_scheduled_tokens,(arm,cap,step)
    for k in ('free_after','preemptions','recomputed_tokens','budget','chunk'):
     assert a.round_rows[-1][k]==b.round_rows[-1][k],(arm,cap,step,k)
    advance(a,x);advance(b,y)
    if not a.requests:break
   assert not a.requests and not b.requests
   if spec['method']=='past_future_v1':assert list(a.predictor.history)==list(b.predictor.history)
   n+=1
 return n

def replay(cell,arm,dt=.05):
 spec=ARMS[arm];reqs=read(OUT/f"inputs/{cell['case']}.json");s=scheduler(blocks=cell['capacity'],cls=MatrixScheduler)
 clock=[0.];s.arrival_clock=lambda:clock[0];s.max_model_len=8176
 context(s,cell['source'],0);s.configure(spec['method'],**spec['config'])
 cursor=0;events=[];targets={r['id']:r['output'] for r in reqs};action=hashlib.sha256()
 bins=dict(crowded=0,decode_dominant=0,prefill_dominant=0,recomputing=0);start=time.monotonic()
 first={};finished={}
 for step in range(100000):
  while cursor<len(reqs) and reqs[cursor]['arrival_ms']/1000<=clock[0]:
   r=reqs[cursor];s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,SamplingParams(max_tokens=r['output']),None,None,arrival_time=clock[0]))
   events.append((clock[0],r['prompt']));cursor+=1
  if not s.requests:
   if cursor==len(reqs):break
   clock[0]=reqs[cursor]['arrival_ms']/1000;continue
  out=s.schedule();row=s.round_rows[-1];state=row['state'];cu=cell['capacity']-1
  assert row['free_before']+sum(r['blocks'] for r in state)==cu
  assert 0<=row['free_after']<=cu
  assert sum(out.num_scheduled_tokens.values())<=row['budget']
  held=sum(r['blocks'] for r in state if r['outputs']>0)
  bins['crowded']+=row['free_before']/cu<.1
  bins['decode_dominant']+=held/cu>.5
  bins['prefill_dominant']+=sum(r['blocks'] for r in state if r['outputs']==0)/cu>.5
  bins['recomputing']+=row['recomputed_tokens']>0
  if spec['method']=='threshold':
   rate=sum(p*math.exp(-(clock[0]-t)) for t,p in events)
   assert math.isclose(rate,row['features']['arrival_rate'],rel_tol=1e-9,abs_tol=1e-8)
   work=sum(max(0,r['prompt']-r['computed']) for r in state if r['outputs']==0)
   n=max(1,min(100000,math.ceil(8+128*row['free_before']/(1+(work+spec['config']['threshold']['tau_s']*rate)/16))))
   assert row['threshold']==n
   if row['intervened']:assert row['baseline_growth_blocks']-(row['free_before']-row['free_after'])>=n
  action.update(json.dumps(out.num_scheduled_tokens,sort_keys=True).encode());done=[]
  for rid in out.num_scheduled_tokens:
   r=s.requests[rid]
   if r.num_computed_tokens==r.num_tokens:r.append_output_token_ids(100);first.setdefault(rid,step)
   if len(r.output_token_ids)==targets[rid]:done.append(rid);finished[rid]=step
  s.finish_requests(done,RequestStatus.FINISHED_LENGTH_CAPPED);clock[0]+=dt
  assert time.monotonic()-start<300,('CPU replay watchdog',cell,arm)
 assert cursor==len(reqs) and not s.requests and len(finished)==len(reqs),(cell,arm)
 return dict(**cell,arm=arm,dt=dt,passed=True,rounds=len(s.round_rows),action_sha256=action.hexdigest(),
  state_coverage=bins,sums={k:sum(r.get(k,0) for r in s.round_rows) for k in ('preemptions','recomputed_tokens','intervened','gate_rejected')},
  elapsed_s=time.monotonic()-start,first_round=first,finish_round=finished,note='Artificial clock; no GPU timing inference.')

def main(full=False):
 d=read(OUT/'design.json');assert d['arms']==ARMS
 if full:cells=d['cells'];target=OUT/'simulator/full';status=OUT/'simulator/status.json';equal=None
 else:
  equal=equivalence();print('Differential trajectories:',equal,flush=True)
  cells=[next(c for c in d['cells'] if c['capacity']==cap and c['source']==['code','conversation','burstchat','burstgpt4'][i%4] and c['window']=='w0' and c['shape']=='accelerating') for i,cap in enumerate(CAPS)]
  cells += [c for c in d['cells'] if c['family']=='synthetic' and c['capacity']==512]
  target=OUT/'simulator/preflight';status=OUT/'simulator/preflight_status.json'
 results=[];expected=len(cells)*len(ARMS)
 for cell in cells:
  for arm in ARMS:
   key=f"{cell['case']}_c{cell['capacity']}_{arm}";p=target/(key+'.json')
   if p.exists():r=read(p);assert r['sources']==source_hashes()
   else:r=replay(cell,arm);r['sources']=source_hashes();write(p,r)
   assert r['passed'];results.append(r)
   write(status,dict(state='running',completed=len(results),expected=expected,last=key,updated=time.time()))
   print(len(results),'/',expected,key,r['rounds'],flush=True)
 if full:
  sensitivity=[]
  for cell in [c for c in cells if c['family']=='synthetic' and c['capacity'] in (512,4096)]:
   for arm in ('old','arrival','past_future_common'):
    for dt in (.02,.1):
     r=replay(cell,arm,dt);sensitivity.append(r);write(OUT/f"simulator/sensitivity/{cell['case']}_c{cell['capacity']}_{arm}_{dt}.json",r)
  write(OUT/'simulator/full_complete.json',dict(passed=True,loops=len(results),sensitivity_loops=len(sensitivity),rounds=sum(r['rounds'] for r in results),sources=source_hashes()))
 else:write(OUT/'cpu_validation.json',dict(passed=True,equivalence=equal,loops=len(results),rounds=sum(r['rounds'] for r in results),sources=source_hashes(),design_sha256=digest(OUT/'design.json')))
 write(status,dict(state='complete',completed=len(results),expected=expected,updated=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');a=p.parse_args();main(a.full)
