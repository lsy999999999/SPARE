"""Exploratory C replay using the exact GPU C policy and native allocator."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from unittest.mock import patch
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import statistics as st
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GPU = ROOT / 'artifacts/www_plan_abc_20261008_r1/c'
OLD = ROOT / 'artifacts/www_evaluation_completion_20261003_r1'
OUT = ROOT / 'artifacts/www_c_simulation_20261008_r1'
PROFILES = ('resource', 'resource_staircase', 'gpu_fitted')


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(p)


def csvwrite(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def replay(cell, profile, arm, options, dest, full_trace=False):
    from research.www_evaluation_completion_20261003 import simulate as sim
    from research.www_evaluation_completion_20261003.timing import FittedTiming
    from research.www_plan_abc_20261008.c_scheduler import CScheduler
    fitted = FittedTiming(read(OLD/'timing_profile.json')) if profile == 'gpu_fitted' else None
    with patch.object(sim, 'FormulaScheduler', CScheduler), patch.object(sim, 'BudgetScheduler', CScheduler):
        return sim.simulate(read(GPU/cell['input']), arm='srp' if arm == 'srp' else 'arrival',
            capacity=cell['capacity'], budget=cell['budget'], chunk=cell['chunk'],
            profile='resource' if fitted else profile, fitted_timing=fitted,
            policy_options=options, output=dest, full_trace=full_trace)


def validate(design):
    import vllm
    from research.www_evaluation_completion_20261003 import simulate as sim
    from research.www_evaluation_completion_20261003.timing import FittedTiming
    native = read(ROOT/'artifacts/www_evaluation_gpu_completion_20261003_r1/manifest.json')['native_vllm_hashes']
    for name, expected in native.items():
        assert digest(Path(vllm.__file__).parent/name) == expected, name
    old = read(OLD/'runs/core__code_new0_burst_stop__c768__gpu_fitted__spare/result.json')
    archived = sim.simulate(read(OLD/'inputs/code_new0_burst_stop.json'), arm='arrival', capacity=768,
        fitted_timing=FittedTiming(read(OLD/'timing_profile.json')), output=OUT/'validation/archived')
    assert archived['trajectory_sha256'] == old['trajectory_sha256']
    checks = []
    for cell in design['cells']:
        if cell['stage'] != 'test' or cell['shape'] != 'burst_stop' or not cell['window'].endswith('test0'):
            continue
        group = read(GPU/'groups'/f"{cell['key']}.json")
        folder = GPU/group['folder']
        for arm, options in [('srp', {}), ('fixed_selected', dict(mode='fixed', fixed_budget=256)),
                             ('guarded_selected', dict(mode='fixed_guarded', fixed_budget=256))]:
            dest = OUT/'validation'/f"{cell['key']}__{arm}"
            result = replay(cell, 'gpu_fitted', arm, options, dest, True)
            gpu_run = read(folder/'runs'/f'{arm}_r0.json')
            assert digest(folder/gpu_run['rounds_path']) == gpu_run['rounds_sha256']
            with gzip.open(folder/gpu_run['rounds_path'], 'rt') as f: gpu_rows = json.load(f)
            with gzip.open(dest/'iterations.jsonl.gz', 'rt') as f: sim_rows = [json.loads(l) for l in f]
            assert len(gpu_rows) == len(sim_rows), (cell['key'], arm, len(gpu_rows), len(sim_rows))
            for i, (g, s) in enumerate(zip(gpu_rows, sim_rows)):
                for field in ('scheduled', 'free_before', 'free_after', 'preemptions', 'recomputed_tokens'):
                    assert g[field] == s[field], (cell['key'], arm, i, field)
            checks.append(dict(cell=cell['key'], arm=arm, matched_rounds=len(sim_rows)))
    # C SPARE is identical to the archived default policy on this frozen replay.
    from research.www_plan_abc_20261008.c_scheduler import CScheduler
    with patch.object(sim, 'FormulaScheduler', CScheduler), patch.object(sim, 'BudgetScheduler', CScheduler):
        exact = sim.simulate(read(OLD/'inputs/code_new0_burst_stop.json'), arm='arrival', capacity=768,
            fitted_timing=FittedTiming(read(OLD/'timing_profile.json')), output=OUT/'validation/exact_c_spare')
    assert exact['trajectory_sha256'] == archived['trajectory_sha256']
    cell = next(c for c in design['cells'] if c['key'] == 'conversation_test1_steady_c2048')
    r0 = replay(cell, 'gpu_fitted', 'srp', {}, OUT/'validation/aa0')
    r1 = replay(cell, 'gpu_fitted', 'srp', {}, OUT/'validation/aa1')
    assert r0['trajectory_sha256'] == r1['trajectory_sha256']
    write(OUT/'validation.json', dict(passed=True, native_hashes=native, burst_checks=checks,
        archived_spare_reproduced=True, exact_c_spare_matches=True,
        deterministic_missing_cell_aa=True, aa_ttft_ms=r0['metrics']['ttft_ms_mean']))


def task(cell, profile, arm, options):
    return dict(cell=cell, profile=profile, arm=arm, options=options)


def execute(t):
    cell=t['cell']; dest=OUT/'runs'/f"{cell['key']}__{t['profile']}__{t['arm']}"
    try:
        cached = read(dest/'result.json') if (dest/'result.json').exists() else None
        if cached and cached.get('c_task') == t and cached.get('complete'):
            r=cached
        else:
            r=replay(cell, t['profile'], t['arm'], t['options'], dest)
            r.update(c_task=t, c_design_sha256=digest(OUT/'design.json'),
                timing_profile_sha256=digest(OLD/'timing_profile.json'),
                active_scheduler='research.www_plan_abc_20261008.c_scheduler.CScheduler')
            write(dest/'result.json', r)
        return dict(stage=cell['stage'], cell=cell['key'], window=cell['window'], shape=cell['shape'],
            capacity=cell['capacity'], profile=t['profile'], arm=t['arm'], **r['metrics'])
    except Exception:
        error=dict(task=t, error=traceback.format_exc())
        write(dest/'error.json', error)
        raise


def run_tasks(pool, tasks, stage):
    rows=[]
    futures=[pool.submit(execute,t) for t in tasks]
    for f in as_completed(futures):
        rows.append(f.result())
        write(OUT/'status.json', dict(stage=stage, completed=len(rows), expected=len(tasks)))
        if len(rows)%24 == 0:print(stage,len(rows),'/',len(tasks),flush=True)
    return sorted(rows,key=lambda r:(r['profile'],r['cell'],r['arm']))


def main():
    design=read(GPU/'design.json')
    assert digest(GPU/'design.json') == read(GPU/'manifest.json')['design_sha256']
    for c in design['cells']:assert digest(GPU/c['input'])==c['input_sha256']
    from research.www_evaluation_completion_20261003.simulate import SOURCE_FILES
    sources=set(SOURCE_FILES) | {'research/www_plan_abc_20261008/c_policy.py',
        'research/www_plan_abc_20261008/c_scheduler.py',
        'research/www_evaluation_gpu_completion_20261003/policy.py',
        'research/www_evaluation_gpu_completion_20261003/scheduler.py',
        'research/www_evaluation_completion_20261003/timing.py',
        str(Path(__file__).relative_to(ROOT))}
    frozen=dict(created=time.time(), gpu_design_sha256=digest(GPU/'design.json'),
        profiles=list(PROFILES), cells=design['cells'], budgets=design['calibration_budgets'],
        gpu_selection=read(GPU/'selection.json')['fixed'],
        source_sha256={p:digest(ROOT/p) for p in sorted(sources)},
        protocol_sha256=digest(Path(__file__).with_name('PROTOCOL_CN.md')),
        timing_profile_sha256=digest(OLD/'timing_profile.json'))
    if (OUT/'design.json').exists():
        prior=read(OUT/'design.json'); frozen['created']=prior['created']; assert frozen==prior
    else:write(OUT/'design.json',frozen)
    validate(design)
    print('Validated archived SPARE, native source hashes, GPU burst allocation sequences, deterministic A/A.',flush=True)
    cal=[task(c,p,f'{family}_{b}',dict(mode='fixed' if family=='fixed' else 'fixed_guarded',fixed_budget=b))
         for c in design['cells'] if c['stage']=='cal' for p in PROFILES
         for family in ('fixed','guarded') for b in design['calibration_budgets']]
    with ProcessPoolExecutor(max_workers=4) as pool:
        cal_rows=run_tasks(pool,cal,'calibration');csvwrite(OUT/'calibration.csv',cal_rows)
        selection={}
        for p in PROFILES:
            scores={f'{fam}_{b}':st.fmean(r['ttft_ms_mean'] for r in cal_rows
                if r['profile']==p and r['arm']==f'{fam}_{b}')
                for fam in ('fixed','guarded') for b in design['calibration_budgets']}
            selection[p]={fam:min(design['calibration_budgets'],key=lambda b:(scores[f'{fam}_{b}'],b))
                          for fam in ('fixed','guarded')}
            selection[p]['scores']=scores
        write(OUT/'selection.json',selection)
        print('Frozen simulation selections', {p:{k:selection[p][k] for k in ('fixed','guarded')} for p in PROFILES},flush=True)
        tests=[]
        for c in design['cells']:
            if c['stage']!='test':continue
            for p in PROFILES:
                opts={'srp':{},'spare':{},'budget_only':dict(candidate_axes='budget')}
                for fam in ('fixed','guarded'):
                    mode='fixed' if fam=='fixed' else 'fixed_guarded'
                    opts[f'{fam}_selected']=dict(mode=mode,fixed_budget=selection[p][fam])
                    if selection[p][fam]!=256:opts[f'{fam}_gpu256']=dict(mode=mode,fixed_budget=256)
                tests.extend(task(c,p,a,o) for a,o in opts.items())
        test_rows=run_tasks(pool,tests,'test');csvwrite(OUT/'test.csv',test_rows)
    by={(r['profile'],r['cell'],r['arm']):r for r in test_rows}
    accepted={r['cell'] for r in csv.DictReader((GPU/'analysis/partial_cell_means.csv').open())}
    pairs=[]
    for c in design['cells']:
        if c['stage']!='test':continue
        for p in PROFILES:
            for ref in ('srp','fixed_selected','guarded_selected','budget_only','fixed_gpu256','guarded_gpu256'):
                actual=ref
                if ref.endswith('gpu256') and selection[p][ref.split('_')[0]]==256:actual=ref.replace('gpu256','selected')
                a,b=by[p,c['key'],'spare'],by[p,c['key'],actual]
                pairs.append(dict(profile=p,cell=c['key'],window=c['window'],shape=c['shape'],capacity=c['capacity'],reference=ref,
                    gpu_accepted=c['key'] in accepted,ttft_reduction_pct=100*(1-a['ttft_ms_mean']/b['ttft_ms_mean']),
                    e2e_reduction_pct=100*(1-a['e2e_ms_mean']/b['e2e_ms_mean'])))
    csvwrite(OUT/'paired.csv',pairs)
    summary=[]
    for p in PROFILES:
        for scope in ('full24','gpu_accepted23'):
            for ref in ('srp','fixed_selected','guarded_selected','budget_only','fixed_gpu256','guarded_gpu256'):
                grp=[r for r in pairs if r['profile']==p and r['reference']==ref and (scope=='full24' or r['gpu_accepted'])]
                v=[r['ttft_reduction_pct'] for r in grp]
                summary.append(dict(profile=p,scope=scope,reference=ref,n=len(v),ttft_reduction_pct=st.fmean(v),
                    e2e_reduction_pct=st.fmean(r['e2e_reduction_pct'] for r in grp),wins=sum(x>1e-9 for x in v),
                    ties=sum(abs(x)<=1e-9 for x in v),losses=sum(x < -1e-9 for x in v)))
    write(OUT/'summary.json',dict(complete=True,calibration_runs=len(cal_rows),test_runs=len(test_rows),
        total_runs=len(cal_rows)+len(test_rows),selection=selection,summary=summary))
    write(OUT/'status.json',dict(stage='complete',completed=len(cal_rows)+len(test_rows),failed=0))
    print('COMPLETE',len(cal_rows)+len(test_rows),flush=True)
    for r in summary:
        if r['scope']=='full24':print(r,flush=True)


if __name__=='__main__':main()
