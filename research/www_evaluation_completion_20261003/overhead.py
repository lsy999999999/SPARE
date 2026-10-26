"""CPU-only policy cost, frozen-state equivalence and decode-guard redundancy."""
import gzip
import json
import statistics
import time
from .common import OUT, write
from .policy import select
from .timing import gpu_runs, round_path
from research.v1_budget_20260922.policy import State


def benchmark():
    samples=[]
    for run in gpu_runs():
        if run['arm']!='arrival' or int(run['repeat'])!=0:continue
        p,_=round_path(run)
        with gzip.open(p,'rt') as h:rows=json.load(h)
        # Uniform time-index sampling, independent of performance or intervention.
        for i in sorted({int(j*(len(rows)-1)/31) for j in range(32)}):
            r=rows[i]
            samples.append((r,[State(**x) for x in r['state']]))
    timings={'audited_all_candidates':[],'early_exit_no_diagnostics':[]}
    changed_decode=changed_finish=early_mismatch=0;counts={k:[] for k in timings}
    for record,rows in samples:
        args=(rows,1024,256,record['free_before'],record['total_blocks'],16,32,8176)
        recent=record['features'];a=select(*args,recent=recent)
        b=select(*args,recent=recent,options={'early_exit':True,'diagnostics':False})
        signature=lambda x:(x['budget'],x['chunk'],None if x['selected'] is None else x['selected'].tokens)
        early_mismatch+=signature(a)!=signature(b)
        changed_decode+=signature(a)!=signature(select(*args,recent=recent,options={'protect_decode':False}))
        changed_finish+=signature(a)!=signature(select(*args,recent=recent,options={'protect_finish':False}))
        counts['audited_all_candidates'].append(a['candidates']);counts['early_exit_no_diagnostics'].append(b['candidates'])
        for repeat in range(6):
            order=list(timings) if repeat%2==0 else list(reversed(timings))
            for name in order:
                opt={} if name=='audited_all_candidates' else {'early_exit':True,'diagnostics':False}
                start=time.perf_counter_ns();select(*args,recent=recent,options=opt)
                timings[name].append((time.perf_counter_ns()-start)/1000)
    assert early_mismatch==0
    from simulator_core.types import percentile
    write(OUT/'overhead.json',dict(complete=True,sampled_states=len(samples),early_exit_action_mismatches=early_mismatch,
          removing_decode_guard_changed_actions=changed_decode,removing_finish_guard_changed_actions=changed_finish,
          costs={k:dict(mean_us=statistics.fmean(v),p50_us=percentile(v,50),p95_us=percentile(v,95),
                        mean_candidates=statistics.fmean(counts[k])) for k,v in timings.items()},
          scope='Single-process CPU pure-policy microbenchmark, alternating order, six measurements per state. Excludes queue sorting, state extraction, native scheduler, file logging and GPU execution. Not production overhead.',
          sample_scope='32 uniformly spaced iterations from each accepted r3 arrival repeat 0. Not weighted by online state frequency.'))
