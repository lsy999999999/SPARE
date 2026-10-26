"""Report repeat variability without censoring or changing the registered decision."""
import gzip,json,hashlib,sys,statistics as st
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_candidate_transfer_20260928.common import *

def main():
    output=[]
    for folder in sorted((OUT/'groups').glob('*')):
        if not (folder/'selected.json').exists():continue
        for t in read(folder/'selected.json')['states']:
            for a in t['actions']:
                ps=[folder/f'runs/s{t["round"]}_transfer_{a["id"]}_r{r}.json' for r in range(REPEATS)]
                if not all(p.exists() for p in ps):continue
                runs=[read(p) for p in ps];rr=[];digests=[]
                for r in runs:
                    p=OUT/r['rounds_path'];assert digest(p)==r['rounds_sha256']
                    with gzip.open(p,'rt') as f:rows=json.load(f)
                    rr.append(rows)
                    physical=[{k:x[k] for k in ('scheduled','free_before','free_after','preemptions','recomputed_tokens')} for x in rows]
                    digests.append(hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest())
                vals=[r['label']['area_per_request_ms'] for r in runs]
                spikes=[]
                for r in range(1,REPEATS):
                    if digests[r]!=digests[0]:continue
                    candidates=[dict(round=i,delta_step_ms=y['step_ms']-x['step_ms'],
                                     reference_step_ms=x['step_ms'],step_ms=y['step_ms'],
                                     reference_schedule_ms=x['schedule_ms'],schedule_ms=y['schedule_ms'])
                                for i,(x,y) in enumerate(zip(rr[0],rr[r])) if i>=t['round']]
                    spikes.append(dict(repeat=r,largest_step_increase=max(candidates,key=lambda x:x['delta_step_ms'])))
                output.append(dict(group=folder.name,target=t['round'],action=a['id'],values=vals,
                    range_ms=max(vals)-min(vals),physical_trajectory_digests=digests,
                    all_physical_equal=len(set(digests))==1,spikes=spikes))
    write(OUT/'analysis/variability.json',dict(note='Descriptive only; no outlier removal, no retesting',actions=output))
    print('Repeat trajectory variability checked:',len(output))

if __name__=='__main__':main()
