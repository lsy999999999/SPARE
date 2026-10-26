"""Select disjoint windows and earlier predictor history before running any timing."""
import csv,math,sys,time
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_arrival_kv_papers_20260925.common import *

def eligible(p,o):return p>0 and 0<o<=512 and p+o<=8000

def main():
 assert not (OUT/'design.json').exists()
 seen=set(); prior_paths=[]
 for directory in (ROOT/'artifacts').glob('v1_*'):
  for name in ('inputs','raw_inputs'):
   for p in (directory/name).glob('*.json'):
    if OUT in p.parents:continue
    data=read(p)
    if isinstance(data,list):
     seen.update(r['id'] for r in data if isinstance(r,dict) and 'id' in r)
     prior_paths.append(str(p.relative_to(ROOT)))
 bases={}; histories={}; provenance={}
 dates={'code':['2024-05-12 10:37','2024-05-14 10:37','2024-05-16 10:37'],
        'conversation':['2024-05-14 10:37','2024-05-15 10:37','2024-05-17 10:37']}
 for source,anchors in dates.items():
  path=ROOT/f'data/azure_{"conv" if source=="conversation" else "code"}_1week.csv'
  chosen={a:[] for a in anchors}; hist=[]; skipped={a:0 for a in anchors}
  with path.open(newline='') as f:
   reader=csv.reader(f);next(reader)
   for i,r in enumerate(reader):
    t=r[0]; p,o=int(r[1]),int(r[2]);rid=f'{source}:raw:{i}'
    if len(hist)<500 and t<anchors[0] and eligible(p,o):
     hist.append(dict(id=rid,output=o,raw_time=datetime.fromisoformat(t).timestamp()))
    a=t[:16]
    if a not in chosen or len(chosen[a])>=64:continue
    if not eligible(p,o) or rid in seen:skipped[a]+=1;continue
    chosen[a].append(dict(id=rid,prompt=p,output=o,raw_time=datetime.fromisoformat(t).timestamp(),raw_row=i))
    if len(hist)==500 and all(len(v)==64 for v in chosen.values()):break
  assert len(hist)==500 and all(len(v)==64 for v in chosen.values()),(source,{a:len(v) for a,v in chosen.items()})
  for w,a in enumerate(anchors):bases[source,f'w{w}']=chosen[a]
  histories[source]=dict(lengths=[r['output'] for r in hist],records=hist)
  provenance[source]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),anchors=anchors,filtered=skipped)
  print('Extracted',source,flush=True)
 path=ROOT/'data/burstgpt/BurstGPT_1.csv';bands=[(86400,172800),(259200,345600),(518400,604800)]
 chosen={(s,f'w{w}'):[] for s in ('burstchat','burstgpt4') for w in range(3)}
 hist={s:[] for s in ('burstchat','burstgpt4')};bad=0
 with path.open(newline='') as f:
  for i,r in enumerate(csv.DictReader(f)):
   try:t=float(r['Timestamp']);p=int(r['Request tokens']);o=int(r['Response tokens'])
   except (ValueError,TypeError,KeyError):bad+=1;continue
   if r['Model'] not in ('ChatGPT','GPT-4') or r['Log Type']!='Conversation log' or not eligible(p,o):continue
   s='burstchat' if r['Model']=='ChatGPT' else 'burstgpt4';rid=f'{s}:raw:{i}'
   if t<86400 and len(hist[s])<500:hist[s].append(dict(id=rid,output=o,raw_time=t))
   for w,(lo,hi) in enumerate(bands):
    if lo<=t<hi and len(chosen[s,f'w{w}'])<64 and rid not in seen:
     chosen[s,f'w{w}'].append(dict(id=rid,prompt=p,output=o,raw_time=t,raw_row=i))
 assert all(len(v)==64 for v in chosen.values())
 bases.update(chosen)
 for s,rs in hist.items():
  assert len(rs)==500
  histories[s]=dict(lengths=[r['output'] for r in rs],records=rs)
 provenance['burstgpt']=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),bands=bands,malformed=bad)
 ids=set();cells=[];cases={}
 for (source,window),rows in bases.items():
  these={r['id'] for r in rows}
  assert len(these)==64 and not these & (seen|ids)
  assert not these & {r['id'] for r in histories[source]['records']}
  assert max(r['raw_time'] for r in histories[source]['records'])<min(r['raw_time'] for r in rows)
  ids|=these;write(OUT/f'raw_inputs/{source}_{window}.json',rows)
  for shape in ('steady','accelerating','burst_stop','trace_relative'):
   case=f'{source}_{window}_{shape}';shaped=[]
   for i,r in enumerate(rows):
    u=i/63;span=max(1e-9,rows[-1]['raw_time']-rows[0]['raw_time'])
    t={'steady':12000*u,'accelerating':4000*(math.sqrt(1+15*u)-1),'burst_stop':0.,
       'trace_relative':12000*(r['raw_time']-rows[0]['raw_time'])/span}[shape]
    shaped.append(dict(id=r['id'],prompt=r['prompt'],output=r['output'],arrival_ms=t))
   assert all(b['arrival_ms']>=a['arrival_ms'] for a,b in zip(shaped,shaped[1:]))
   cases[case]=shaped
   for cap in CAPS:cells.append(dict(case=case,source=source,window=window,shape=shape,capacity=cap,family='measured_lengths'))
 old=read(ROOT/'artifacts/v1_broad12h_20260925_r1/design.json')
 for c in old['cells']:
  if c['family']!='synthetic' or c['capacity']!=512:continue
  cases[c['case']]=read(ROOT/f"artifacts/v1_broad12h_20260925_r1/inputs/{c['case']}.json")
  for cap in CAPS:cells.append({k:v for k,v in dict(c,capacity=cap).items() if k!='stage'})
 histories['synthetic']=dict(lengths=[histories[s]['lengths'][i] for i in range(125) for s in dates.keys()|{'burstchat','burstgpt4'}],
                            note='Fixed balanced earlier measured history; no synthetic evaluation outputs used.')
 # Sort source iteration explicitly for a platform-independent predictor input.
 histories['synthetic']['lengths']=[histories[s]['lengths'][i] for i in range(125) for s in sorted(histories) if s!='synthetic']
 write(OUT/'histories.json',histories)
 for case,rs in cases.items():write(OUT/f'inputs/{case}.json',rs)
 design=dict(frozen=True,design_frozen_at=time.time(),methods=list(ARMS),arms=ARMS,cells=cells,capacities=CAPS,repeats=4,
  expected_formal_runs=len(cells)*len(ARMS)*4,trace_hashes={k:object_digest(v) for k,v in cases.items()},
  histories_sha256=digest(OUT/'histories.json'),provenance=provenance,previous_input_files=prior_paths,
  distinct_new_requests=len(ids),deadline_hours=96,episode_timeout_s=600,max_group_retries=1,
  gpu_preference=[6,7,0,1,2,3],max_model_len=8176,configs=BASE,
  objective='Mean TTFT; equal cell weighting after averaging four repeats. No performance-dependent stopping or parameter tuning.',
  scope='Only old+A is the proposed policy; no D variants. Paper-rule adaptations, not original serving systems.',
  notes='12 disjoint measured cohorts; shapes and capacities reuse each cohort. 4 synthetic controls separate. GPU timing only; simulator artificial clocks.')
 write(OUT/'design.json',design)
 print('Frozen',len(cells),'groups',design['expected_formal_runs'],'GPU runs',len(ids),'new requests')
if __name__=='__main__':main()
