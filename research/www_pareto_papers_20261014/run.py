"""CPU-only extension of the frozen Pareto pilot to all available paper adapters."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch
from research.www_pareto_20261014.run import ROOT, BUDGETS, CAPACITIES, read, write, digest

PREVIOUS = ROOT / 'artifacts/www_pareto_20261014_r1'
OUT = ROOT / 'artifacts/www_pareto_papers_20261014_r1'
MAIN = ('arrival', 'native', 'sarathi', 'past_future', 'past_future_multi', 'orca_max',
        's3_proxy', 'ltr_proxy', 'apt_serve_kv_rule', 'sola_rule', 'sola_calibrated', 'uniboost_static')
DIAGNOSTIC = ('ssjf_oracle', 'trail_oracle', 'mc_sf_oracle', 'tie_oracle_proxy', 'ltr_oracle')
ARMS = MAIN + DIAGNOSTIC
LEGACY = ('orca_max', 's3_proxy', 'ltr_proxy', 'ltr_oracle')
ENHANCED = ('past_future_multi', 'sola_calibrated')
LABELS = dict(arrival='SPARE', native='vLLM V1', sarathi='Sarathi rule',
              past_future='Past-Future rule', past_future_multi='Past-Future multi',
              orca_max='Orca reservation', s3_proxy='S3 packing proxy', ltr_proxy='LTR score proxy',
              apt_serve_kv_rule='Apt-Serve KV rule', sola_rule='SOLA risk rule',
              sola_calibrated='SOLA calibrated', uniboost_static='UniBoost rule',
              ssjf_oracle='SSJF oracle', trail_oracle='TRAIL oracle', mc_sf_oracle='MC-SF oracle',
              tie_oracle_proxy='TIE oracle-proxy', ltr_oracle='LTR oracle')
SCOPES = dict(
    arrival='Current SPARE, unchanged; no hidden output lengths.',
    native='Modern vLLM 0.9.2 V1; engineering representative, not original SOSP 2023 kernels.',
    sarathi='Decode/partial-prefill/FIFO rule on V1 incremental KV; no original kernels or pipeline.',
    past_future='History-conditional length sampling and future peak admission; shared V1 backend.',
    past_future_multi='32 draws for batch<16, mean peak gate; declared local choices, not author defaults.',
    orca_max='FIFO iteration scheduling and prompt+visible 2048-output-token reservation; shared chunking, no selective batching.',
    s3_proxy='10 output buckets, historical median, first-fit packing and reservation doubling; no DistilBERT or swap.',
    ltr_proxy='Historical median score and starvation boost; no OPT/ListMLE predictor or cross-queue swap.',
    apt_serve_kv_rule='Pending/KV density, beta=0 special case; no hidden cache or full phase optimization.',
    sola_rule='Historical risk-ordering approximation with fixed internal cost formula and SLOs.',
    sola_calibrated='Calibrated phase costs, online EMA feedback, phase/budget choices; shared V1 constraints; SLO 5s/200ms.',
    uniboost_static='Phase signal, geometric quantization, fixed gamma and priority victim replacement; recomputation replaces swap.',
    ssjf_oracle='True output length replaces BERT predictor; diagnostic only.',
    trail_oracle='True remaining length and limited preemption; recomputation replaces swap; diagnostic only.',
    mc_sf_oracle='True output length, shortest-first and future KV peak admission; diagnostic only.',
    tie_oracle_proxy='True length supplies log-t mean with fixed sigma; no DeBERTa predictor; diagnostic only.',
    ltr_oracle='True output length supplies score; no OPT/ListMLE or cross-queue swap; diagnostic only.')


def initialize():
    global SIM, TIMING, HISTORY, COST, ORIGINAL_MAKE, LEGACY_CLASS, ENHANCED_CLASS
    from research.www_evaluation_completion_20261003 import simulate as sim
    from research.www_evaluation_completion_20261003.timing import FittedTiming
    from research.www_legacy_baselines_20261006.scheduler import GPUBaselineScheduler as Legacy
    from research.www_baseline_fidelity_20261013.scheduler import GPUBaselineScheduler as Enhanced
    SIM, ORIGINAL_MAKE = sim, sim.make_scheduler
    LEGACY_CLASS, ENHANCED_CLASS = Legacy, Enhanced
    sim.digest = lru_cache(maxsize=None)(sim.digest)
    TIMING = FittedTiming(read(PREVIOUS / 'timing_profile.json'))
    HISTORY = {key: [r['output'] for r in rows] for key, rows in read(PREVIOUS / 'history.json').items()}
    COST = read(ROOT / 'artifacts/www_baseline_fidelity_20261013_r1/calibration/cost_model.json')


def simulate(requests, arm, capacity, budget, history, seed):
    capture = {}
    def make(arm='arrival', blocks=768, budget=1024, chunk=256, **kwargs):
        if arm in LEGACY + ENHANCED:
            cls = LEGACY_CLASS if arm in LEGACY else ENHANCED_CLASS
            with patch.object(SIM, 'BudgetScheduler', cls):
                scheduler, clock = ORIGINAL_MAKE('native', blocks, budget, chunk, **kwargs)
            scheduler.history_context = kwargs.get('history', ())
            scheduler.seed_context = kwargs.get('seed', 0)
            scheduler.oracle_lengths_context = kwargs.get('oracle_lengths') or {}
            scheduler.fidelity_cost_model = COST
            scheduler.configure(arm, budget, chunk, kwargs.get('concurrency', 32))
        else:
            scheduler, clock = ORIGINAL_MAKE(arm, blocks, budget, chunk, **kwargs)
        capture['scheduler'] = scheduler
        return scheduler, clock
    class Clock:
        def estimate(self, batch):
            duration, details = TIMING.estimate(batch)
            # The enhanced SOLA controller reads this completed step at the next round.
            capture['scheduler'].round_rows[-1]['step_ms'] = duration
            return duration, details
    adaptation = dict(SIM.ADAPTATION, **{a: ('paper_adapter', (SCOPES[a],)) for a in ARMS})
    with patch.object(SIM, 'make_scheduler', make), \
         patch.object(SIM, 'ARMS', tuple(set(SIM.ARMS) | set(ARMS))), \
         patch.object(SIM, 'ORACLE_ARMS', tuple(set(SIM.ORACLE_ARMS) | set(DIAGNOSTIC))), \
         patch.object(SIM, 'ADAPTATION', adaptation):
        result = SIM.simulate(requests, arm=arm, capacity=capacity, budget=budget, chunk=256,
                              history=history, seed=seed, fitted_timing=Clock(), output=None)
    result['timing_claim'] = 'CPU simulation using frozen retrospective fitted clock; no GPU execution'
    result['adapter_scope'] = SCOPES[arm]
    scheduler = capture['scheduler']
    result['feedback_gamma'] = scheduler.cost.gamma if arm == 'sola_calibrated' else None
    return result


def prepare():
    if (OUT / 'design.json').exists():
        return read(OUT / 'design.json')
    previous = read(PREVIOUS / 'design.json')
    for rel, sha in previous['hashes'].items():
        assert digest(ROOT / rel) == sha, rel
    hashes = dict(previous['hashes'])
    sources = [f'research/{folder}/{name}' for folder, names in (
        ('www_pareto_papers_20261014', ('run.py',)),
        ('www_legacy_baselines_20261006', ('scheduler.py', 'policy.py')),
        ('www_baseline_fidelity_20261013', ('scheduler.py', 'policy.py')),
        ('www_evaluation_gpu_completion_20261003', ('scheduler.py', 'policy.py')),
    ) for name in names]
    for rel in sources:
        hashes[rel] = digest(ROOT / rel)
    for rel in ('artifacts/www_baseline_fidelity_20261013_r1/calibration/cost_model.json',
                'artifacts/www_pareto_20261014_r1/design.json'):
        hashes[rel] = digest(ROOT / rel)
    for rel in hashes:
        if rel.endswith('.py'):
            dst = OUT / 'source' / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((ROOT / rel).read_bytes())
    design = dict(previous, arms=ARMS, main_arms=MAIN, diagnostic_arms=DIAGNOSTIC,
                  expected_runs=len(previous['samples']) * len(CAPACITIES) * len(ARMS) * len(BUDGETS),
                  hashes=hashes, adapter_scopes=SCOPES,
                  excluded={'srp': 'Internal ablation, excluded from external comparison at user request.',
                            'TokenFlow': 'No client consumption/buffer trace or GPU-CPU cache migration model.'},
                  fairness='Four budgets per implementation; duplicate paper variants are also reported under their common family.',
                  scope='13 external paper/system families, 16 external implementations plus SPARE. '
                        'Oracle diagnostics kept separate from visible-input rule/proxy comparison.')
    write(OUT / 'design.json', design)
    return design


def validate():
    initialize()
    previous = read(PREVIOUS / 'design.json')
    checks = []
    for source in ('code', 'conversation'):
        sample = next(s for s in previous['samples'] if s['source'] == source and s['split'] == 'validation')
        reqs = read(ROOT / sample['input'])
        seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
        refs = {(r['arm'], r['budget']): r for r in read(PREVIOUS / 'runs' / f"{sample['name']}_c768.json")}
        for arm in ARMS:
            start = time.monotonic()
            result = simulate(reqs, arm, 768, 1024, HISTORY[source], seed)
            m = result['metrics']
            assert m['completed'] == len(reqs)
            assert m['executed_tokens'] - m['recomputed_tokens'] == sum(r['prompt'] + r['output'] - 1 for r in reqs)
            if (arm, 1024) in refs:
                assert result['trajectory_sha256'] == refs[arm, 1024]['trajectory_sha256'], arm
            if arm == 'sola_calibrated':
                assert result['feedback_gamma'] > 0 and result['feedback_gamma'] != 1
            check = dict(source=source, arm=arm, passed=True, cpu_s=time.monotonic()-start,
                         feedback_gamma=result['feedback_gamma'])
            checks.append(check)
            print(json.dumps(check), flush=True)
    write(OUT / 'validation.json', dict(passed=True, checks=checks, previous_trajectory_parity=True,
                                       conservation=True, sola_online_feedback=True))


def run_task(task):
    sample, cap, arm = task
    dest = OUT / 'runs' / f"{sample['name']}_c{cap}_{arm}.json"
    if dest.exists():
        rows = read(dest)
        if len(rows) == 4 and all(r['complete'] for r in rows):
            return 4
    reqs = read(ROOT / sample['input'])
    seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
    rows = []
    for budget in BUDGETS:
        result = simulate(reqs, arm, cap, budget, HISTORY[sample['source']], seed)
        m = result['metrics']
        assert m['completed'] == m['requests'] == 64
        assert abs(m['output_throughput_tps'] - sum(r['output'] for r in reqs)*1000/m['elapsed_simulated_ms']) < 1e-9
        rows.append(dict(**sample, capacity=cap, arm=arm, budget=budget, complete=True,
                         seed=seed, metrics=m, trajectory_sha256=result['trajectory_sha256'],
                         diagnostic=arm in DIAGNOSTIC, adapter_scope=SCOPES[arm],
                         feedback_gamma=result['feedback_gamma']))
    write(dest, rows)
    return 4


def run(jobs):
    d = prepare()
    assert read(OUT / 'validation.json')['passed']
    for rel, sha in d['hashes'].items():
        assert digest(ROOT / rel) == sha, rel
    tasks = [(s, c, a) for s in d['samples'] for c in CAPACITIES for a in ARMS]
    start, completed, failures = time.monotonic(), 0, []
    with ProcessPoolExecutor(max_workers=jobs, initializer=initialize) as pool:
        futures = {pool.submit(run_task, t): (t[0]['name'], t[1], t[2]) for t in tasks}
        for future in as_completed(futures):
            try:
                completed += future.result()
            except Exception as exc:
                failure = dict(task=futures[future], error=repr(exc))
                failures.append(failure)
                print('FAILED', failure, flush=True)
            if completed % 100 == 0 or not future.exception() is None:
                status = dict(completed=completed, expected=d['expected_runs'], failures=failures,
                              elapsed_s=time.monotonic()-start)
                write(OUT / 'status.json', status)
                print(json.dumps(status), flush=True)
    write(OUT / 'status.json', dict(completed=completed, expected=d['expected_runs'], failures=failures,
                                   elapsed_s=time.monotonic()-start))
    assert not failures and completed == d['expected_runs'], failures


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('prepare', 'validate', 'run'))
    p.add_argument('--jobs', type=int, default=12)
    args = p.parse_args()
    if args.command == 'prepare':
        print(prepare()['expected_runs'])
    elif args.command == 'validate':
        validate()
    else:
        run(args.jobs)
