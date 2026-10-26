"""Read-only GPU A/A diagnosis plus frozen local clock perturbation."""
from pathlib import Path
from unittest.mock import patch
import gzip
import json
import statistics as st
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.www_c_simulation_20261008.run import GPU, OUT, OLD, read, write, digest, csvwrite


def gpu_diagnosis():
    failed=[]
    for attempt in range(3):
        folder=GPU/'attempts/conversation_test1_steady_c2048'/f'a{attempt}'
        runs=[read(folder/'runs'/f'aa_srp_r{i}.json') for i in range(2)]
        rounds=[]
        for run in runs:
            assert digest(folder/run['rounds_path'])==run['rounds_sha256']
            with gzip.open(folder/run['rounds_path'],'rt') as f:rounds.append(json.load(f))
        k=next(i for i,(a,b) in enumerate(zip(*rounds)) if a['scheduled']!=b['scheduled'])
        difference=set(rounds[0][k]['scheduled']) ^ set(rounds[1][k]['scheduled'])
        details=[]
        for j,run in enumerate(runs):
            row=rounds[j][k]; start=row['step_end_ms']-row['step_ms']
            steps=sum(r['step_ms'] for r in rounds[j])
            details.append(dict(repeat=j,gpu=run['gpu'],ttft_ms=run['metrics']['ttft_ms_mean'],
                elapsed_ms=run['elapsed_ms'],step_sum_ms=steps,outside_step_ms=run['elapsed_ms']-steps,
                rounds=len(rounds[j]),preemptions=run['metrics']['preemptions'],recomputed_tokens=run['metrics']['recomputed_tokens'],
                divergent_start_ms=start,scheduled=row['scheduled'],
                differing_requests=[q for q in run['requests'] if q['request_id'] in difference]))
        failed.append(dict(attempt=attempt,first_divergent_round=k,aa=read(folder/'aa.json'),runs=details))
    accepted=[]
    for p in sorted((GPU/'groups').glob('*.json')):
        g=read(p); folder=GPU/g['folder']
        r=[read(folder/'runs'/f'aa_srp_r{i}.json') for i in range(2)]
        accepted.append(dict(cell=g['cell'],stage=g['stage'],gpu=g['gpu'],
            second_ttft_change_pct=100*(r[1]['metrics']['ttft_ms_mean']/r[0]['metrics']['ttft_ms_mean']-1),
            second_elapsed_change_pct=100*(r[1]['elapsed_ms']/r[0]['elapsed_ms']-1)))
    write(OUT/'gpu_aa_diagnosis.json',dict(failed=failed,accepted=accepted,
        accepted_groups=len(accepted),second_slower=sum(r['second_ttft_change_pct']>0 for r in accepted),
        median_second_ttft_change_pct=st.median(r['second_ttft_change_pct'] for r in accepted)))


class ScaledTiming:
    def __init__(self, timing, scale):self.timing,self.scale=timing,scale
    def estimate(self,batch):
        duration,flags=self.timing.estimate(batch)
        return duration*self.scale,flags


def main():
    from research.www_evaluation_completion_20261003 import simulate as sim
    from research.www_evaluation_completion_20261003.timing import FittedTiming
    from research.www_plan_abc_20261008.c_scheduler import CScheduler
    design=read(GPU/'design.json')
    cell=next(c for c in design['cells'] if c['key']=='conversation_test1_steady_c2048')
    scales=[.98,.99,1.,1.01,1.02]
    opts={'srp':{},'spare':{},'fixed256':dict(mode='fixed',fixed_budget=256),
          'guarded256':dict(mode='fixed_guarded',fixed_budget=256),'budget_only':dict(candidate_axes='budget')}
    write(OUT/'diagnostic_design.json',dict(cell=cell,scales=scales,options=opts,
        protocol_sha256=digest(Path(__file__).with_name('DIAGNOSTIC_PROTOCOL_CN.md')),
        runner_sha256=digest(Path(__file__)),timing_sha256=digest(OLD/'timing_profile.json')))
    gpu_diagnosis()
    rows=[]
    for scale in scales:
        for arm,options in opts.items():
            dest=OUT/'clock_sensitivity'/f'{scale:.2f}__{arm}'
            with patch.object(sim,'FormulaScheduler',CScheduler),patch.object(sim,'BudgetScheduler',CScheduler):
                r=sim.simulate(read(GPU/cell['input']),arm='srp' if arm=='srp' else 'arrival',capacity=2048,
                    fitted_timing=ScaledTiming(FittedTiming(read(OLD/'timing_profile.json')),scale),
                    policy_options=options,output=dest)
            r.update(clock_scale=scale,c_task_arm=arm,diagnostic_design_sha256=digest(OUT/'diagnostic_design.json'))
            write(dest/'result.json',r)
            rows.append(dict(scale=scale,arm=arm,ttft_ms=r['metrics']['ttft_ms_mean'],
                e2e_ms=r['metrics']['e2e_ms_mean'],preemptions=r['metrics']['preemptions'],
                recomputed_tokens=r['metrics']['recomputed_tokens'],rounds=r['metrics']['rounds']))
        print('Clock scale',scale,'complete',flush=True)
    csvwrite(OUT/'clock_sensitivity.csv',rows)
    write(OUT/'clock_sensitivity.json',dict(complete=True,runs=25,rows=rows))


if __name__=='__main__':main()
