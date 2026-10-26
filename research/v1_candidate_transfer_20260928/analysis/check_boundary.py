"""Supplementary boundary contrasts already present among preselected arms.

No new runs; filter on action structure, never on measured sign. This was not a
preregistered primary hypothesis and cannot replace the full hardware sweep.
"""
import gzip,json,statistics as st,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_candidate_transfer_20260928.common import *

def main():
    output=[]
    for folder in sorted((OUT/'groups').glob('*')):
        if not (folder/'selected.json').exists():continue
        for t in read(folder/'selected.json')['states']:
            current=next(a for a in t['actions'] if a['id']==t['current_id'])
            for a in t['actions']:
                if not (current['q'] in (128,256,512) and 0<a['q']-current['q']<=2 and
                        a['chunk']==current['chunk'] and a['growth']==current['growth']):continue
                differences={rid:a['tokens'].get(rid,0)-current['tokens'].get(rid,0)
                             for rid in a['tokens'].keys()|current['tokens'].keys()}
                if min(differences.values())<0:continue
                allruns=[];allrows=[]
                for act in (current,a):
                    ps=[folder/f'runs/s{t["round"]}_transfer_{act["id"]}_r{r}.json' for r in range(REPEATS)]
                    if not all(p.exists() for p in ps):break
                    rs=[read(p) for p in ps];rr=[]
                    for r in rs:
                        p=OUT/r['rounds_path'];assert digest(p)==r['rounds_sha256']
                        with gzip.open(p,'rt') as f:rr.append(json.load(f)[t['round']])
                    allruns.append(rs);allrows.append(rr)
                if len(allruns)!=2:continue
                times=[[x['step_ms']-x['schedule_ms'] for x in rr] for rr in allrows]
                gains=[l['label']['area_per_request_ms']-r['label']['area_per_request_ms'] for l,r in zip(*allruns)]
                output.append(dict(group=folder.name,round=t['round'],current_id=current['id'],other_id=a['id'],
                    q=[current['q'],a['q']],chunk=a['chunk'],growth=a['growth'],
                    token_differences={rid:v for rid,v in differences.items() if v},
                    scheduled_request_counts=[current['scheduled_requests'],a['scheduled_requests']],
                    non_schedule_ms=times,ratios=[y/x for x,y in zip(*times)],mean_ratio=st.fmean(times[1])/st.fmean(times[0]),
                    current_minus_other_wait_ms=gains,
                    first_wait_gain_ms=[l['label']['first_round_ms']-r['label']['first_round_ms'] for l,r in zip(*allruns)],
                    later_wait_gain_ms=[l['label']['later_ms']-r['label']['later_ms'] for l,r in zip(*allruns)]))
    write(OUT/'analysis/boundary_reference.json',dict(note='Supplementary descriptive contrasts, structural filter; not predeclared primary endpoint',contrasts=output))
    for c in output:print(c['group'],c['round'],c['q'],round(100*(c['mean_ratio']-1),2),[round(v,2) for v in c['current_minus_other_wait_ms']])

if __name__=='__main__':main()
