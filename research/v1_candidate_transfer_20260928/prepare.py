"""Select disjoint windows using source timestamps only; freeze before GPU labels."""
import csv, hashlib, shutil, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_candidate_transfer_20260928.common import *

def main():
    assert not (OUT/'design.json').exists()
    seen=set(); scanned=[]
    for directory in sorted((ROOT/'artifacts').glob('*/*inputs')):
        if directory.parent==OUT: continue
        for p in sorted(directory.glob('*.json')):
            data=read(p)
            if isinstance(data,list):
                seen.update(r['id'] for r in data if isinstance(r,dict) and 'id' in r)
                scanned.append(str(p.relative_to(ROOT)))
    dates={'code':['2024-05-12 13:41','2024-05-15 13:41'],
           'conversation':['2024-05-14 13:41','2024-05-17 13:41']}
    groups=[]; provenance={}; hashes={}; allids=set()
    for source,minutes in dates.items():
        path=ROOT/f'data/azure_{"conv" if source=="conversation" else "code"}_1week.csv'
        chosen={m:[] for m in minutes}; skipped={m:dict(ineligible=0,overlap=0) for m in minutes}
        with path.open(newline='') as f:
            reader=csv.reader(f); next(reader)
            for i,row in enumerate(reader):
                m=row[0][:16]
                if m not in chosen or len(chosen[m])==64:continue
                p,o=int(row[1]),int(row[2]); rid=f'{source}:raw:{i}'
                if not (p>0 and 0<o<=512 and p+o<=8000):skipped[m]['ineligible']+=1;continue
                if rid in seen:skipped[m]['overlap']+=1;continue
                chosen[m].append(dict(id=rid,prompt=p,output=o,raw_timestamp=row[0],raw_row=i))
                if all(len(v)==64 for v in chosen.values()):break
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        provenance[source]=dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest(),anchors=minutes,skipped=skipped)
        for w,m in enumerate(minutes):
            raw=chosen[m];assert len(raw)==64,(source,m,len(raw))
            ids={r['id'] for r in raw};assert len(ids)==64 and not ids&seen and not ids&allids
            allids.update(ids);name=f'{source}_fresh{w}'
            write(OUT/f'raw_inputs/{name}.json',raw)
            reqs=[dict(id=r['id'],prompt=r['prompt'],output=r['output'],arrival_ms=0.) for r in raw]
            write(OUT/f'inputs/{name}.json',reqs);hashes[name]=digest(OUT/f'inputs/{name}.json')
            for b,k in CONFIGS:
                cap=768 if source=='conversation' else 1536
                groups.append(dict(key=f'{name}_c{cap}_t{b}_k{k}',source=source,window=w,input=name,capacity=cap,budget=b,chunk=k))
    d=dict(frozen_at=time.time(),groups=groups,input_hashes=hashes,source_hashes=sources(),
           provenance=provenance,overlap_audit=dict(files=scanned,previous_unique_ids=len(seen),new_ids=len(allids),overlap=0),
           repeats=REPEATS,claim='Preselected single-action mechanism transfer; not closed-loop gain')
    write(OUT/'design.json',d)
    for rel in d['source_hashes']:
        dest=OUT/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    print(dict(groups=len(groups),new_requests=len(allids),previous_ids=len(seen)),flush=True)

if __name__=='__main__':main()
