"""Post-run descriptive decomposition; does not select or change experimental arms."""
import gzip,json,statistics as st,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_candidate_transfer_20260928.common import *

def rows(run):
    p=OUT/run['rounds_path'];assert digest(p)==run['rounds_sha256']
    with gzip.open(p,'rt') as f:return json.load(f)

def main():
    summary=read(OUT/'analysis/summary.json');descriptions=[]
    for c in summary['contrasts']:
        folder=OUT/'groups'/c['group'];index=c['round']
        anchor=rows(read(folder/'runs/anchor.json'));state=anchor[index]['state']
        runs={aid:[read(folder/f'runs/s{index}_transfer_{aid}_r{r}.json') for r in range(REPEATS)]
              for aid in (c['left'],c['right'])}
        target=next(t for t in read(folder/'selected.json')['states'] if t['round']==index)
        current=[read(folder/f'runs/s{index}_transfer_{target["current_id"]}_r{r}.json') for r in range(REPEATS)]
        vs_current={}
        for aid,rs in runs.items():
            gains=[a['label']['area_per_request_ms']-b['label']['area_per_request_ms'] for a,b in zip(current,rs)]
            vals=[[a['label']['area_per_request_ms'] for a in arr] for arr in (current,rs)]
            noise=max(1.,*(max(v)-min(v) for v in vals))
            vs_current[aid]=dict(gains_ms=gains,mean_gain_ms=st.fmean(gains),noise_scale_ms=noise,
                positive_screen=all(g>noise for g in gains),negative_screen=all(g< -noise for g in gains),
                note='Supplementary descriptive reference contrast; not the primary predeclared K comparison')
        actions={aid:rs[0]['action'] for aid,rs in runs.items()}
        changed=[]
        for i,s in enumerate(state):
            lt=actions[c['left']]['tokens'].get(s['rid'],0);rt=actions[c['right']]['tokens'].get(s['rid'],0)
            if lt!=rt:changed.append(dict(queue_index=i,**s,left_tokens=lt,right_tokens=rt,
                                         remaining=s['tokens']-s['computed']))
        request_deltas={s['rid']:[] for s in state if s['outputs']==0}
        branchrows={aid:[rows(r)[index] for r in rs] for aid,rs in runs.items()}
        for repeat in range(REPEATS):
            residual={}
            for aid,rs in runs.items():
                r=rs[repeat];start=(branchrows[aid][repeat]['probe_decision_wall']-r['start_wall'])*1000
                residual[aid]={rid:r['clocks'][rid]['first_ms']-start for rid in request_deltas}
            for rid in request_deltas:
                request_deltas[rid].append(residual[c['left']][rid]-residual[c['right']][rid])
        # Reconstruct aggregate label using request-level paired waits.
        for repeat in range(REPEATS):
            reconstructed=sum(v[repeat] for v in request_deltas.values())/64
            assert abs(reconstructed-c['gains_ms'][repeat])<1e-6
        means={rid:st.fmean(v) for rid,v in request_deltas.items()}
        descriptions.append(dict(group=c['group'],round=index,kind=c['kind'],n=c['n'],
            actual_current=c['actual_current'],changed_requests=changed,
            current_id=target['current_id'],vs_current=vs_current,
            pending=len(request_deltas),left=actions[c['left']],right=actions[c['right']],
            positive_contribution_ms=sum(v for v in means.values() if v>0)/64,
            negative_contribution_ms=sum(v for v in means.values() if v<0)/64,
            per_request_delta_ms={rid:dict(mean_ms=means[rid],repeats_ms=v) for rid,v in request_deltas.items()},
            left_step_ms=[r['step_ms'] for r in branchrows[c['left']]],
            right_step_ms=[r['step_ms'] for r in branchrows[c['right']]],
            left_schedule_ms=[r['schedule_ms'] for r in branchrows[c['left']]],
            right_schedule_ms=[r['schedule_ms'] for r in branchrows[c['right']]],
            left_preemptions=[r['label']['preemptions'] for r in runs[c['left']]],
            right_preemptions=[r['label']['preemptions'] for r in runs[c['right']]],
            left_recomputed=[r['label']['recomputed_tokens'] for r in runs[c['left']]],
            right_recomputed=[r['label']['recomputed_tokens'] for r in runs[c['right']]]))
    write(OUT/'analysis/explanation.json',dict(note='Post-run descriptive, no outcome-based arm selection',contrasts=descriptions))
    print('Reconstructed request-level contributions:',len(descriptions))

if __name__=='__main__':main()
