"""Post-hoc descriptive examples; no additional performance experiments."""
import collections,gzip,statistics as st,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_ratio_closed_loop_20260928.common import *
from research.v1_budget_20260922.policy import State
from research.v1_broad12h_20260925.policy import select as legacy
targets=[('code_fresh1_c1536_t1024_k256','steady','ratio16'),
         ('code_fresh0_c1536_t1024_k1024','steady','ratio4'),
         ('conversation_fresh0_c768_t1024_k128','steady','ratio4')]
examples=[]
for group,shape,arm in targets:
    folder=OUT/'groups'/group
    runs=[]
    for mode in ('fixed64',arm):
        for repeat in (0,1):
            r=read(folder/'runs'/f'{shape}_{mode}_r{repeat}.json');cfg=r['config']
            with gzip.open(OUT/r['rounds_path'],'rt') as f:rows=json.load(f)
            counter=collections.Counter();deltas=collections.Counter();first=None
            for row in rows:
                ss=[State(**s) for s in row['state']]
                ref=legacy(ss,cfg['budget'],cfg['chunk'],row['free_before'],row['total_blocks'],16,32,8176,
                           cfg['threshold'],dict(arrival_rate=row['features']['arrival_rate']))
                p=ref['selected']
                if p is None or p.tokens==row['scheduled']:continue
                counter['changed']+=1
                if all(s.outputs==0 or s.decode for s in ss):counter['without_recovery_request']+=1
                if not any(s.outputs>0 for s in ss):counter['without_any_output_started']+=1
                for s in ss:
                    delta=row['scheduled'].get(s.rid,0)-p.tokens.get(s.rid,0)
                    deltas['pending' if s.outputs==0 else 'output_started']+=delta
                if first is None:
                    first=dict(round=row['round'],free=row['free_before'],threshold=row['threshold'],
                        actual_q=sum(row['scheduled'].values()),reference_q=sum(p.tokens.values()),
                        actual_growth=row['actual_growth_blocks'],reference_growth=p.growth,
                        actual_parameters=[row['budget'],row['chunk']],reference_parameters=[ref['budget'],ref['chunk']],
                        allocations=[dict(id=s.rid,remaining=s.remaining,outputs=s.outputs,computed=s.computed,blocks=s.blocks,
                            actual=row['scheduled'].get(s.rid,0),reference=p.tokens.get(s.rid,0)) for s in ss if s.rid in p.tokens or s.rid in row['scheduled']])
            runs.append(dict(arm=mode,repeat=repeat,metrics=r['metrics'],counts=dict(counter),deltas=dict(deltas),first_changed=first,
                actual_q_mean=st.fmean(sum(row['scheduled'].values()) for row in rows)))
    examples.append(dict(group=group,shape=shape,arm=arm,runs=runs))
write(OUT/'analysis/mechanism_examples.json',dict(post_hoc_descriptive=True,examples=examples))
for e in examples:
    print(e['group'],e['arm'])
    for r in e['runs']:
        print(r['arm'],r['repeat'],{k:r['metrics'][k] for k in ('mean_ttft_ms','round_count','preemptions','recomputed_tokens','schedule_ms_total','step_ms_total')},r['counts'],r['deltas'])
        if r['first_changed']:print('first_changed',r['first_changed'])
