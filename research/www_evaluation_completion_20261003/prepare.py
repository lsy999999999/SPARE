"""Freeze inputs and every experimental cell before observing new outcomes."""
import csv
import json
import time
from collections import Counter
from .common import ROOT, HERE, OUT, GPU, read, write, digest

PROFILES = ['resource', 'resource_staircase', 'gpu_fitted']
CORE = {
    'spare': {}, 'basic': {'tau':0}, 'srp': {'engine_arm':'srp'},
    'fixed256': {'mode':'fixed','fixed_budget':256},
    'fixed512': {'mode':'fixed','fixed_budget':512},
    'occupancy70': {'mode':'occupancy','occupancy':.7,'fixed_budget':512},
    'constant16': {'mode':'constant','n':16},
    'constant32': {'mode':'constant','n':32},
    'no_finish_guard': {'protect_finish':False},
}
SENSITIVITY = {'n0_4':{'n0':4},'n0_16':{'n0':16},
               'alpha64':{'alpha':64},'alpha256':{'alpha':256},
               'H05':{'H':.5},'H2':{'H':2},'tau05':{'tau':.5},'tau2':{'tau':2}}


def prepare():
    if (OUT/'design.json').exists():return read(OUT/'design.json')
    seen=set(); scanned=[]
    for pattern in ('v1*/*inputs/*.json', 'www*/*inputs/*.json'):
        for p in sorted((ROOT/'artifacts').glob(pattern)):
            if OUT in p.parents:continue
            data=read(p)
            if isinstance(data,list):
                seen.update(str(r['id']) for r in data if isinstance(r,dict) and 'id' in r)
                scanned.append(str(p.relative_to(ROOT)))
    for p in (ROOT/'writing/www/experiments/inputs').glob('*.json'):
        seen.update(str(r['id']) for r in read(p));scanned.append(str(p.relative_to(ROOT)))
    dates={'code':['2024-05-11 16:37','2024-05-13 16:37','2024-05-16 16:37'],
           'conversation':['2024-05-13 16:37','2024-05-15 16:37','2024-05-18 16:37']}
    bases={}; provenance={}; new_ids=set()
    for source,anchors in dates.items():
        p=ROOT/f'data/azure_{"conv" if source=="conversation" else "code"}_1week.csv'
        chosen={a:[] for a in anchors};skip=Counter()
        with p.open(newline='') as h:
            reader=csv.reader(h);next(reader)
            for i,row in enumerate(reader):
                stamp=row[0][:16]
                for a in anchors:
                    if len(chosen[a])>=512 or stamp<a or stamp[:10]!=a[:10]:continue
                    prompt,output=int(row[1]),int(row[2]);rid=f'{source}:raw:{i}'
                    if not (prompt>0 and 0<output<=512 and prompt+output<=8000):
                        skip['ineligible']+=1;continue
                    if rid in seen or rid in new_ids:skip['overlap']+=1;continue
                    chosen[a].append(dict(id=rid,prompt=prompt,output=output,raw_timestamp=row[0]))
                    new_ids.add(rid)
                if all(len(v)==512 for v in chosen.values()):break
        assert all(len(v)==512 for v in chosen.values()),(source,{a:len(v) for a,v in chosen.items()})
        provenance[source]=dict(path=str(p.relative_to(ROOT)),sha256=digest(p),anchors=anchors,skipped=dict(skip))
        for i,a in enumerate(anchors):
            name=f'{source}_new{i}'
            write(OUT/f'raw_inputs/{name}.json',chosen[a]);bases[name]=chosen[a]
        print('Selected',source,'1536 previously unused requests',flush=True)
    assert not new_ids & seen
    tasks=[]; inputs={}
    def input_file(name,rows):
        dest=OUT/f'inputs/{name}.json';write(dest,rows);inputs[name]=digest(dest)
        return str(dest.relative_to(ROOT))
    def add(family,case,input_path,capacity,profile,arm,options,**extra):
        name=f'{family}__{case}__c{capacity}__{profile}__{arm}'
        tasks.append(dict(name=name,family=family,case=case,input=input_path,capacity=capacity,
                          profile=profile,arm=arm,options=options,**extra))
    for name,raw in bases.items():
        for shape in ('steady','burst_stop'):
            # 128 distinct requests, controlled rate 5.25 req/s, never clipped lengths.
            reqs=[dict(id=r['id'],prompt=r['prompt'],output=r['output'],
                       arrival_ms=1000*i/5.25 if shape=='steady' else 0.) for i,r in enumerate(raw[:128])]
            case=f'{name}_{shape}';path=input_file(case,reqs)
            for cap in (768,2048,4096):
                for profile in PROFILES:
                    for arm,opt in CORE.items():
                        add('core',case,path,cap,profile,arm,opt,window=name,shape=shape)
                    if cap in (768,2048):
                        for arm,opt in SENSITIVITY.items():
                            add('sensitivity',case,path,cap,profile,arm,opt,window=name,shape=shape)
        # All six windows: 512 unique arrivals, a controlled load sweep followed by drain.
        for rate in (2.,5.,10.,20.):
            case=f'{name}_rps{rate:g}'
            reqs=[dict(id=r['id'],prompt=r['prompt'],output=r['output'],arrival_ms=1000*i/rate)
                  for i,r in enumerate(raw)]
            path=input_file(case,reqs)
            for cap in (768,2048,4096):
                for profile in PROFILES:
                    for arm in ('spare','basic','srp','fixed512'):
                        add('sustained',case,path,cap,profile,arm,CORE[arm],window=name,shape='steady',arrival_rps=rate)
    # Retrospective end-to-end validation on fresh1, disjoint from fresh0 timing fit.
    suite=read(ROOT/'writing/www/experiments/suite.json')
    excluded={'trail_oracle','uniboost_static'}  # Their GPU and CPU victim semantics differ.
    for c in suite['cases']:
        if 'fresh1' not in c['name'] or c['profile']!='resource':continue
        case=c['name'].removesuffix('_resource')
        for profile in PROFILES:
            for arm in suite['arms']:
                if arm in excluded:continue
                opt={} if arm=='arrival' else {'tau':0} if arm=='basic' else {'engine_arm':arm}
                add('validation',case,c['input'],c['capacity'],profile,arm,opt,history=c['history'])
    assert len({t['name'] for t in tasks})==len(tasks)
    # Alternate families through the queue, so all experiment types start promptly.
    grouped={f:[t for t in tasks if t['family']==f] for f in sorted({t['family'] for t in tasks})}
    ordered=[]
    for i in range(max(map(len,grouped.values()))):
        for group in grouped.values():
            if i<len(group):ordered.append(group[i])
    design=dict(frozen_at=time.time(),tasks=ordered,expected=len(tasks),
                by_family=dict(Counter(t['family'] for t in tasks)),profiles=PROFILES,
                input_hashes=inputs,provenance=provenance,
                overlap_audit=dict(previous_ids=len(seen),new_ids=len(new_ids),overlap=0,scanned=scanned),
                core=CORE,sensitivity=SENSITIVITY,
                scope='Deterministic native V1 CPU simulation; no GPU calls. Two synthetic clocks and one retrospective GPU trace fit.',
                independent_units='Six new source/date windows. Shapes, capacities, rates and profiles reuse their requests; not independent samples.',
                selection='All simple controls are prespecified and reported; no per-test winner selection. No parameter tuning on new outcomes.',
                timing_validation='fresh0 calibration / fresh1 validation, historical windows; not newly collected GPU profiling.',
                overhead='Isolated CPU policy microbenchmark, no CUDA/production overhead claim.',
                limits=dict(max_rounds=500000,max_wall_s=900),
                source_hashes={str(p.relative_to(ROOT)):digest(p) for p in HERE.glob('*.py')})
    write(OUT/'design.json',design)
    for rel in design['source_hashes']:
        dst=OUT/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/rel).read_bytes())
    return design

if __name__=='__main__':
    print(prepare()['by_family'])
