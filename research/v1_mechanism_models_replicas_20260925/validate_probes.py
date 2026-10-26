"""Native CPU identity test for the one-action replay harness."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_mechanism_models_replicas_20260925.common import *


def main():
    from research.v1_budget_20260922.test_scheduler import scheduler,add,advance
    from research.v1_mechanism_models_replicas_20260925.probe_scheduler import ProbeScheduler
    def replay(times=None,target=None,branch=None):
        s=scheduler(blocks=192,cls=ProbeScheduler)
        s.probe_times_context=times;s.probe_target_context=target
        s.configure('threshold',**BASE,threshold=formula(1),branch=branch)
        add(s,[600+i*13 for i in range(20)])
        for _ in range(10000):
            out=s.schedule();advance(s,out)
            if not s.requests:break
        assert not s.requests
        return s.round_rows
    anchor=replay();eligible=[r for r in anchor if r['intervened']];assert eligible
    row=eligible[len(eligible)//2];target=row['round'];times=[r['features']['arrival_time_s'] for r in anchor]
    counts={}
    for arm,spec in [('arrival',formula(1)),('off','off')]:
        rows=replay(times,target,dict(round=target,fingerprint=row['state_fingerprint'],threshold=spec))
        assert all(rows[i]['state_fingerprint']==anchor[i]['state_fingerprint'] for i in range(target+1))
        assert rows[target]['scheduled']==(row['scheduled'] if arm=='arrival' else row['baseline'])
        counts[arm]=len(rows)
    record=dict(passed=True,matched_prefix_rounds=target+1,rounds=counts,physical_saving=row['baseline_growth_blocks']-row['actual_growth_blocks'],
                source_hashes={str((HERE/name).relative_to(ROOT)):digest(HERE/name) for name in ('probe_scheduler.py','probes.py','validate_probes.py')})
    write(OUT/'probes/cpu_validation.json',record);print(record,flush=True)


if __name__=='__main__':main()
