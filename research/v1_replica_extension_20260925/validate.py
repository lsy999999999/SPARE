"""Native allocator equivalence and causal router checks; not GPU timings."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_replica_extension_20260925.common import *


def main():
    from research.v1_arrival_kv_papers_20260925.scheduler import MatrixScheduler
    from research.v1_broad12h_20260925.scheduler import FormulaScheduler
    from research.v1_paper_comparison_20260922.scheduler import ComparisonScheduler,PAPER_METHODS
    from research.v1_budget_20260922.scheduler import BudgetScheduler
    from research.v1_budget_20260922.test_scheduler import scheduler,add,advance
    trajectories=rounds=0
    for arm,spec in ARMS.items():
        cls=FormulaScheduler if spec['method']=='threshold' else ComparisonScheduler if spec['method'] in PAPER_METHODS else BudgetScheduler
        for capacity in (48,96):
            a,b=[scheduler(blocks=capacity,cls=c) for c in (MatrixScheduler,cls)]
            for s in (a,b):
                s.history_context=[24]*1000;s.seed_context=42;s.arrival_clock=lambda:0.
                s.configure(spec['method'],**spec['config']);add(s,[257,129,385,99,400,12])
            for step in range(4000):
                x,y=a.schedule(),b.schedule()
                assert x.num_scheduled_tokens==y.num_scheduled_tokens,(arm,step)
                for key in ('free_after','preemptions','recomputed_tokens','budget','chunk'):
                    assert a.round_rows[-1][key]==b.round_rows[-1][key]
                advance(a,x);advance(b,y);rounds+=1
                if not a.requests:break
            assert not a.requests and not b.requests
            trajectories+=1
    # Partition and weak scaling invariants independently of arrival order/lengths.
    reqs=[dict(id=str(i),prompt=100+i,output=24,arrival_ms=i*10) for i in range(64)]
    for replicas in (1,2,4):
        mapping=routes(reqs,replicas)
        assert list(mapping.values())==[i%replicas for i in range(64)]
        varied=[dict(r,prompt=9999-r['prompt'],output=1000) for r in reqs]
        assert routes(varied,replicas)==mapping
        for i in range(1,65):assert routes(reqs[:i],replicas)==dict(list(mapping.items())[:i])
        copied=[dict(r,id=f"{r['id']}:copy{j}") for r in reqs for j in range(replicas)]
        cm=routes(copied,replicas)
        for rank in range(replicas):
            selected=[{k:v for k,v in r.items() if k!='id'} for r in copied if cm[r['id']]==rank]
            assert selected==[{k:v for k,v in r.items() if k!='id'} for r in reqs]
    from research.v1_replica_extension_20260925.run import summarize
    clocks={'a':dict(arrival_ms=0,submitted_ms=1,first_ms=10,finish_ms=12,tokens=2,token_times_ms=[10,12]),
            'b':dict(arrival_ms=5,submitted_ms=6,first_ms=25,finish_ms=25,tokens=1,token_times_ms=[25])}
    metrics=summarize(clocks,[],25)
    assert metrics['mean_ttft_ms']==15 and metrics['waiting_area_ms']==30 and metrics['throughput_tokens_s']==120
    # Unequal replica counts must not turn the mean into a mean of worker means.
    unequal={str(i):dict(arrival_ms=0,submitted_ms=0,first_ms=t,finish_ms=t,
        tokens=1,token_times_ms=[t],replica=i%4) for i,t in enumerate([10,20,30,40,90])}
    metric=summarize(unequal,[],90)
    assert metric['mean_ttft_ms']==38 and metric['waiting_area_ms']==190
    assert abs(metric['throughput_tokens_s']-5/.09)<1e-10
    result=dict(replica_counts_tested=[1,2,4],passed=True,trajectories=trajectories,rounds=rounds,router_prefix_causal=True,
                weak_scaling_exact_stream_copies=True,global_metrics_verified=True,sources=sources())
    write(OUT/'cpu_validation.json',result)
    print({k:v for k,v in result.items() if k!='sources'},flush=True)


if __name__=='__main__':main()
