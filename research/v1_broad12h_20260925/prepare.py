"""Freeze a broad matrix; no performance-dependent sampling or coefficient fitting."""
import csv,math,sys,time
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_broad12h_20260925.common import *
from research.v1_broad12h_20260925.policy import formula
CAPS=[512,768,1024,1536,2048,3072,4096]

def shaped(rows,shape):
    first=rows[0]['raw_time'];span=max(1e-9,rows[-1]['raw_time']-first);out=[]
    for i,r in enumerate(rows):
        u=i/(len(rows)-1)
        t={'steady':12000*u,'accelerating':4000*(math.sqrt(1+15*u)-1),
           'burst_stop':0.,'trace_relative':12000*(r['raw_time']-first)/span}[shape]
        out.append(dict(id=r['id'],prompt=r['prompt'],output=r['output'],arrival_ms=t))
    return out

def main():
    assert not (OUT/'design.json').exists()
    seen=set()
    for prefix in ('v1_simple_formula','v1_decode_load_formula','v1_constant_shift','v1_arrival_formula','v1_theory_formula','v1_decode_arrival_formula','v1_decode_arrival_confirm','v1_pressure_gate'):
        for directory in (ROOT/'artifacts').glob(prefix+'_*_r1/inputs'):
            for p in directory.glob('*.json'):
                data=read(p)
                if isinstance(data,list):seen.update(r['id'] for r in data if isinstance(r,dict) and 'id' in r)
    provenance={};bases={};cells=[];cases={}
    dates={'code':['2024-05-11 09:17','2024-05-16 19:17'], 'conversation':['2024-05-13 09:17','2024-05-18 19:17']}
    def eligible(prompt,output):return 0<prompt and 0<output<=512 and prompt+output<=8000
    for source,minutes in dates.items():
        path=ROOT/f'data/azure_{"conv" if source=="conversation" else "code"}_1week.csv'
        chosen={m:[] for m in minutes};skipped={m:0 for m in minutes}
        with path.open(newline='') as f:
            reader=csv.reader(f);next(reader)
            for i,r in enumerate(reader):
                m=r[0][:16]
                if m not in chosen or len(chosen[m])>=64:continue
                prompt,output=int(r[1]),int(r[2]);rid=f'{source}:raw:{i}'
                if not eligible(prompt,output) or rid in seen:skipped[m]+=1;continue
                chosen[m].append(dict(id=rid,prompt=prompt,output=output,raw_time=datetime.fromisoformat(r[0]).timestamp(),raw_row=i))
                if all(len(v)==64 for v in chosen.values()):break
        for wi,m in enumerate(minutes):
            assert len(chosen[m])==64,(source,m,len(chosen[m]))
            bases[source,f'w{wi}']=chosen[m]
        provenance[source]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),anchors=minutes,filtered=skipped)
        print('Extracted',source,flush=True)
    path=ROOT/'data/burstgpt/BurstGPT_1.csv';chosen={(s,w):[] for s in ('burstchat','burstgpt4') for w in ('w0','w1')};malformed=0;filtered={k:0 for k in chosen}
    # Fixed source-time bands, not selected by workload or performance.
    bands={'w0':(172800,259200),'w1':(432000,518400)}
    with path.open(newline='') as f:
        for i,r in enumerate(csv.DictReader(f)):
            try:t=float(r['Timestamp']);prompt=int(r['Request tokens']);output=int(r['Response tokens'])
            except (ValueError,TypeError,KeyError):malformed+=1;continue
            if r['Model'] not in ('ChatGPT','GPT-4') or r['Log Type']!='Conversation log':continue
            source='burstchat' if r['Model']=='ChatGPT' else 'burstgpt4'
            for w,(lo,hi) in bands.items():
                if not lo<=t<hi or len(chosen[source,w])>=64:continue
                rid=f'{source}:raw:{i}'
                if not eligible(prompt,output) or rid in seen:filtered[source,w]+=1;continue
                chosen[source,w].append(dict(id=rid,prompt=prompt,output=output,raw_time=t,raw_row=i))
    assert all(len(v)==64 for v in chosen.values())
    bases.update(chosen)
    provenance['burstgpt']=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),bands=bands,malformed_rows=malformed,
        filtered={str(k):v for k,v in filtered.items()},note='Conversation logs only; source model labels are workload provenance, all replay uses Qwen2.5-7B.')
    for (source,window),rows in bases.items():
        assert len({r['id'] for r in rows})==len(rows) and not seen & {r['id'] for r in rows}
        write(OUT/f'raw_inputs/{source}_{window}.json',rows)
        for shape in ('steady','accelerating','burst_stop','trace_relative'):
            case=f'{source}_{window}_{shape}';cases[case]=shaped(rows,shape)
            caps=([512,768,1536,2048,4096] if shape!='trace_relative' else [768,1536,4096])
            if window=='w1' and shape in ('steady','accelerating'):caps+= [1024,3072]
            for cap in caps:
                cells.append(dict(case=case,source=source,window=window,shape=shape,capacity=cap,
                    family='measured_lengths',stage='core' if cap in (512,768,1536,2048,4096) and shape!='trace_relative' else 'extension'))
    # Controlled request compositions, no fabricated performance timing model.
    for scenario in ('prefill_heavy','decode_heavy','mixed_lengths','resident_decode'):
        case='synthetic_'+scenario;rows=[]
        for i in range(48):
            if scenario=='prefill_heavy':prompt,output,t=4096,16,0.
            elif scenario=='decode_heavy':prompt,output,t=256,512,12000*i/47
            elif scenario=='mixed_lengths':prompt,output,t=(256,32,12000*i/47) if i%2==0 else (7936,32,12000*i/47)
            else:prompt,output,t=(2048,512,0.) if i<8 else (1024,32,4000+8000*(i-8)/39)
            rows.append(dict(id=f'{case}:{i}',prompt=prompt,output=output,arrival_ms=t))
        cases[case]=rows
        for cap in (512,768,1536,4096):cells.append(dict(case=case,source='synthetic',window='controlled',shape=scenario,capacity=cap,family='synthetic',stage='stress'))
    for name,rows in cases.items():
        assert max(r['prompt']+r['output'] for r in rows)<=8000
        write(OUT/f'inputs/{name}.json',rows)
    arms=dict(old=formula(0),arrival=formula(1),combined=formula(1,32),pressure_decode=formula(1,32,'pressure_decode'),pressure_constant=formula(1,32,'pressure_constant'))
    design=dict(frozen=True,design_frozen_at=time.time(),methods=list(arms),arms=arms,cells=cells,repeats=2,
        expected_formal_runs=len(cells)*len(arms)*2,capacities=CAPS,trace_hashes={n:object_digest(r) for n,r in cases.items()},
        provenance=provenance,source_families=['code','conversation','burstchat','burstgpt4'],
        eligibility='0 < output <= 512; prompt > 0; prompt+output <= 8000; first 64 eligible requests at predeclared anchor; no token-length clipping.',
        configs=BASE,deadline_hours=12,episode_timeout_s=240,max_group_retries=1,gpu_preference=[7,0,6,1,2,3],
        objective='Mean TTFT only. Fixed formula comparison; no online tuning or selection of coefficients. Synthetic and measured-length results reported separately.',
        arrival_note='All measured-length shape spans normalized to 12s except simultaneous burst_stop. trace_relative preserves relative gaps, not original time scale.',
        scope='One Qwen model, multiple workload provenance/day windows, seven capacities, four arrival shapes, four synthetic KV compositions.',
        execution='Whole source-window-shape-capacity group (five arms x two repeats) on one GPU; deterministic shuffled order; dynamic idle GPU leasing. Deadline leaves explicit incomplete groups.',
        limitation='Synthetic token ID 100 repeated to replay lengths, not original text; all cohorts initially empty. resident_decode induces an evolving resident state, not a forced occupancy.')
    write(OUT/'design.json',design)
    print('Frozen',len(cells),'groups;',design['expected_formal_runs'],'formal runs;',len(cases),'traces')

if __name__=='__main__':main()
