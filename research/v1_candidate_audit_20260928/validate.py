"""Native CPU allocator replay, never GPU timing evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_candidate_audit_20260928.common import *
from research.v1_candidate_audit_20260928.run import select_targets


def main():
    from vllm import SamplingParams
    from vllm.v1.request import Request,RequestStatus
    from research.v1_budget_20260922.test_scheduler import scheduler
    from research.v1_fixed_order_20260927.probe_scheduler import ProbeScheduler
    summaries=[]
    def make(reqs,cap,cfg,times,target):
        s=scheduler(blocks=cap,cls=ProbeScheduler)
        s.probe_enabled_context=True;s.probe_times_context=times;s.probe_target_context=target
        s.configure('threshold',**cfg)
        for i,r in enumerate(reqs):
            s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,SamplingParams(max_tokens=r['output']),None,None,arrival_time=float(i)))
        return s
    def step(s,targets):
        out=s.schedule();done=[]
        for rid in out.num_scheduled_tokens:
            r=s.requests[rid]
            if r.num_computed_tokens==r.num_tokens:r.append_output_token_ids(100)
            if len(r.output_token_ids)==targets[rid]:done.append(rid)
        s.finish_requests(done,RequestStatus.FINISHED_LENGTH_CAPPED)
    for source,cap in [('conversation',768),('code',1536)]:
        reqs=read(OLD/f'inputs/{source}_w0_burst_stop.json');targets={r['id']:r['output'] for r in reqs}
        for b,k in CONFIGS:
            cfg=dict(budget=b,chunk=k,concurrency=32,threshold=spec())
            s=make(reqs,cap,cfg,[i*.05 for i in range(10000)],9999)
            for _ in range(10000):
                step(s,targets)
                if not s.requests:break
            assert not s.requests
            anchor=s.round_rows;selected=select_targets(anchor,b,k);checks=0
            for target in selected['states']:
                index=target['round'];times=[r['features']['arrival_time_s'] for r in anchor]
                for action in target['actions']:
                    branch=dict(round=index,fingerprint=target['fingerprint'],
                        threshold=dict(spec(),forced_action=dict(budget=action['budget'],chunk=action['chunk'])),after_threshold=spec())
                    s=make(reqs,cap,dict(cfg,branch=branch),times,index)
                    for i in range(index+2):
                        step(s,targets);row=s.round_rows[-1]
                        if i<=index:assert row['state_fingerprint']==anchor[i]['state_fingerprint']
                        if i<index:assert row['scheduled']==anchor[i]['scheduled']
                        if i==index:
                            assert row['scheduled']==action['tokens'] and row['branch_applied']
                        checks+=1
                    s.finish_requests(list(s.requests),RequestStatus.FINISHED_ABORTED)
                    assert not s.requests and s.kv_cache_manager.block_pool.get_num_free_blocks()==cap-1
            summary=dict(source=source,capacity=cap,budget=b,chunk=k,anchor_rounds=len(anchor),
                         states=len(selected['states']),actions=sum(len(t['actions']) for t in selected['states']),prefix_checks=checks)
            summaries.append(summary);print(summary,flush=True)
    write(OUT/'validation.json',dict(passed=True,source_hashes=sources(),checks=summaries,
                                  note='Native CPU allocator with artificial prefix clock; no performance labels'))


if __name__=='__main__':main()
