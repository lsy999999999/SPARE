"""Frozen exploratory V1 matrix. Run via bootstrap.sh on the designated GPU."""
from __future__ import annotations
import argparse
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import statistics as st
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('V1_BUDGET_OUTPUT', ROOT/('artifacts/v1_budget_gpu'+os.environ.get('V1_BUDGET_GPU', '4')+'_20260922'))).resolve()
MODEL = Path(os.environ.get('A_V1_MODEL', ROOT/'models/Qwen2.5-7B-Instruct')).resolve()
HERE = Path(__file__).resolve().parent
DATA = ROOT/'research/state_conditioned_prefill_selection_20260918'
CONFIGS = [dict(budget=512, chunk=128, concurrency=16),
           dict(budget=1024, chunk=256, concurrency=32),
           dict(budget=2048, chunk=2048, concurrency=64)]
REGIMES = ('queue', 'burst', 'restricted')
METHODS = ('native', 'srp', 'budget')
ARMS = ('native', 'srp', 'budget', 'srp_matched', 'shadow')


class TuningTimeout(Exception):
    """A recorded, expected configuration failure; restart with a fresh engine."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def hardware():
    gpu = os.environ.get('V1_BUDGET_GPU', '4')
    row = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
          '--query-gpu=name,uuid,driver_version,memory.total',
          '--format=csv,noheader,nounits'], text=True).strip()
    name, uuid, driver, memory = [x.strip() for x in row.split(',')]
    if 'RTX 4090' not in name and 'L20' not in name:
        raise RuntimeError(f'Expected RTX 4090 or L20 at designated GPU {gpu}: {row}')
    return dict(host=socket.gethostname(), gpu=gpu, name=name, uuid=uuid,
                driver=driver, memory_mib=int(memory))


def source_hashes():
    dist = importlib.metadata.distribution('vllm')
    assert dist.version == '0.9.2', dist.version
    files = {str(p.relative_to(ROOT)): p for p in HERE.glob('*.py')}
    for name in ('bootstrap.sh', 'PROTOCOL.md'):
        files[str((HERE/name).relative_to(ROOT))] = HERE/name
    for name in ('requirements-a-v1-gpu.txt', 'prepare_a_v1_transfer.py', 'transfer/model_manifest.json'):
        path = ROOT/'research/simulator_rl_20260920'/name
        files[str(path.relative_to(ROOT))] = path
    files[str((DATA/'data_manifest.json').relative_to(ROOT))] = DATA/'data_manifest.json'
    for name in ('v1/core/sched/scheduler.py', 'v1/core/kv_cache_manager.py',
                 'v1/core/single_type_kv_cache_manager.py', 'v1/core/block_pool.py'):
        files['vllm/'+name] = Path(dist.locate_file('vllm/'+name))
    return {k: digest(v) for k, v in files.items()}


def make_blocks():
    m = read(DATA/'data_manifest.json')
    result = {}
    for source in ('conversation', 'code'):
        for role, split, fractions, count in [('dev', 'train', [.1], 32),
                                              ('compare', 'validation', [.1, .3], 64)]:
            spec = m['sources'][source]['splits'][split]
            path = ROOT/spec['path']
            assert digest(path) == spec['sha256']
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            old = set()
            for fraction in ([.4] if role == 'dev' else [.75, .9]):
                i = int(len(rows)*fraction)
                old.update(r['request_id'] for r in rows[i:i+count])
            seen = set()
            for j, fraction in enumerate(fractions):
                start = int(len(rows)*fraction)
                chosen = rows[start:start+count]
                ids = {r['request_id'] for r in chosen}
                assert len(chosen) == len(ids) == count and not ids & (old | seen)
                seen |= ids
                assert all(0 < r['prompt_tokens'] and 0 < r['actual_output_tokens']
                           and r['prompt_tokens']+4*r['actual_output_tokens'] <= 12288 for r in chosen)
                result[f'{role}_{source}_{j}'] = dict(role=role, source=source,
                    start=start, source_sha256=spec['sha256'], rows=chosen)
    return result


def initialize():
    versions = {p: importlib.metadata.version(p) for p in ('vllm', 'torch', 'transformers', 'numpy')}
    manifest = dict(schema=2, implementation='V1-budget-light-1/timeout-scoring-2',
                    hardware=hardware(), model=str(MODEL), versions=versions,
                    model_revision=read(ROOT/'research/simulator_rl_20260920/transfer/model_manifest.json')['revision'],
                    source_hashes=source_hashes(), blocks=make_blocks(), configs=CONFIGS,
                    regimes=REGIMES, methods=METHODS, arms=ARMS, repeats=3,
                    restricted_blocks=768, timeout_s=300, aa_max_range_over_mean=.20,
                    claim='exploratory disjoint blocks from previously used source minutes; not independent-day confirmation',
                    engine=dict(v1=True, multiprocessing=False, eager=True, prefix_cache=False,
                                block_size=16, max_model_len=12288, gpu_memory_utilization=.9))
    # JSON normalization makes immutable tuples compare consistently on resume.
    manifest = json.loads(json.dumps(manifest))
    target = OUT/'manifest.json'
    if target.exists():
        if read(target) != manifest:
            raise RuntimeError('Manifest changed: use a NEW output directory; old results remain frozen')
    else:
        write(target, manifest)
        for p in HERE.glob('*'):
            if p.is_file():
                target = OUT/'source'/p.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(p.read_bytes())
    return manifest


def requests_for(block, regime):
    rows = block['rows'][:24 if block['role']=='compare' else 12] if regime=='queue' else block['rows']
    first = rows[0]['original_arrival_ms']
    span = max(1., rows[-1]['original_arrival_ms']-first)
    return [dict(id=r['request_id'], prompt=r['prompt_tokens'],
                 output=r['actual_output_tokens']*(1 if regime=='queue' else 4),
                 arrival_ms=(r['original_arrival_ms']-first)/span*(2000 if regime=='queue' else 20))
            for r in rows]


def jobs(manifest, phase, regime, selection=None):
    role = 'compare' if phase=='compare' else 'dev'
    blocks = [k for k, v in manifest['blocks'].items() if v['role']==role]
    result = []
    for rep in range(3 if phase in ('aa', 'compare') else 1):
        for bi, block in enumerate(blocks):
            arms = list(ARMS if phase=='compare' else METHODS if phase=='tune' else ('srp',))
            shift = (bi+rep) % len(arms)
            arms = arms[shift:]+arms[:shift]
            if rep % 2:
                arms.reverse()
            for ci in range(3 if phase=='tune' else 1):
                for arm in arms:
                    cfg = CONFIGS[ci] if phase=='tune' else CONFIGS[1]
                    if phase=='compare':
                        cfg = selection[regime]['budget' if arm in ('shadow', 'srp_matched') else arm]
                    key = f'{phase}_{regime}_{block}_{arm}_c{ci}_r{rep}'
                    result.append(dict(key=key, phase=phase, regime=regime, block=block,
                                       arm=arm, method='srp' if arm=='srp_matched' else arm,
                                       config=cfg, repeat=rep))
    return result


def quantile(values, q):
    if not values:
        return None
    xs = sorted(values)
    p = (len(xs)-1)*q
    i = int(p)
    return xs[i]+(xs[min(i+1, len(xs)-1)]-xs[i])*(p-i)


def build_engine(regime):
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    expected_uuid = subprocess.check_output(['nvidia-smi',
        '--id='+os.environ.get('V1_BUDGET_GPU', '4'), '--query-gpu=uuid',
        '--format=csv,noheader'], text=True).strip().removeprefix('GPU-').lower()
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid).lower()
    assert actual_uuid == expected_uuid, ('CUDA/NVML GPU mismatch', actual_uuid, expected_uuid)
    capacity_path = OUT/'capacity.json'
    capacity = read(capacity_path)['blocks'] if capacity_path.exists() else None
    assert capacity is not None or regime == 'smoke'
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=str(MODEL), dtype='half', max_model_len=12288,
        max_num_seqs=64, max_num_batched_tokens=2048,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        block_size=16, gpu_memory_utilization=.9, enforce_eager=True,
        disable_log_stats=True, seed=0,
        num_gpu_blocks_override=768 if regime=='restricted' else capacity,
        scheduler_cls='research.v1_budget_20260922.scheduler.BudgetScheduler'))
    scheduler = engine.engine_core.engine_core.scheduler
    actual = scheduler.kv_cache_config.num_blocks
    if regime != 'restricted':
        if capacity is None:
            write(capacity_path, dict(blocks=actual))
        else:
            assert actual == capacity
    else:
        assert actual == 768
    return engine, scheduler


def episode(engine, scheduler, reqs, method, config, timeout=300):
    from vllm import SamplingParams
    scheduler.configure(method, **config)
    clocks = {r['id']: dict(arrival_ms=r['arrival_ms'], first_ms=None,
                          finish_ms=None, tokens=0, token_times_ms=[]) for r in reqs}
    cursor = 0
    error = None
    start = time.perf_counter()
    try:
        while cursor < len(reqs) or engine.has_unfinished_requests():
            now = (time.perf_counter()-start)*1000
            if now > timeout*1000:
                raise TimeoutError('Predeclared episode deadline exceeded')
            while cursor < len(reqs) and reqs[cursor]['arrival_ms'] <= now:
                r = reqs[cursor]
                clocks[r['id']]['submitted_ms'] = now
                engine.add_request(r['id'], {'prompt_token_ids': [100]*r['prompt']},
                    SamplingParams(temperature=0., max_tokens=r['output'], ignore_eos=True,
                                   detokenize=False))
                cursor += 1
            if engine.has_unfinished_requests():
                step_start = time.perf_counter()
                outputs = engine.step()
                now = (time.perf_counter()-start)*1000
                scheduler.round_rows[-1]['step_ms'] = (time.perf_counter()-step_start)*1000
                for out in outputs:
                    c = clocks[out.request_id]
                    n = len(out.outputs[0].token_ids) if out.outputs else 0
                    if n > c['tokens']:
                        assert n == c['tokens']+1, 'Expected synchronous single-token output'
                        c['tokens'] = n
                        c['token_times_ms'].append(now)
                        if c['first_ms'] is None:
                            c['first_ms'] = now
                    if out.finished:
                        c['finish_ms'] = now
            else:
                time.sleep(min(.005, max(0, (reqs[cursor]['arrival_ms']-now)/1000)))
        assert all(clocks[r['id']]['tokens']==r['output'] and clocks[r['id']]['finish_ms'] is not None for r in reqs)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    elapsed = (time.perf_counter()-start)*1000
    rounds = scheduler.round_rows
    complete = sum(clocks[r['id']]['finish_ms'] is not None and clocks[r['id']]['tokens']==r['output'] for r in reqs)
    ttft = [c['first_ms']-c['arrival_ms'] for c in clocks.values() if c['first_ms'] is not None]
    tbt = [b-a for c in clocks.values() for a, b in zip(c['token_times_ms'], c['token_times_ms'][1:])]
    metrics = dict(completion_rate=complete/len(reqs),
                   mean_ttft_ms=st.fmean(ttft) if error is None else None,
                   p95_ttft_ms=quantile(ttft, .95) if error is None else None,
                   mean_tbt_ms=st.fmean(tbt) if tbt else None, p95_tbt_ms=quantile(tbt, .95),
                   throughput_tokens_s=sum(c['tokens'] for c in clocks.values())/(elapsed/1000),
                   preemptions=sum(r['preemptions'] for r in rounds),
                   recomputed_tokens=sum(r['recomputed_tokens'] for r in rounds),
                   interventions=sum(r['intervened'] for r in rounds),
                   candidates=sum(r['candidates'] for r in rounds),
                   verified_rounds=sum(r['verified'] for r in rounds),
                   recovery_fallbacks=sum(r['recovery_fallback'] for r in rounds),
                   peak_kv_fraction=max((1-r['free_after']/r['total_blocks'] for r in rounds), default=0),
                   round_count=len(rounds))
    for field in ('decision_ms', 'native_ms', 'audit_ms', 'schedule_ms', 'step_ms'):
        xs = [r[field] for r in rounds if field in r]
        metrics[field+'_mean'] = st.fmean(xs) if xs else None
        metrics[field+'_p95'] = quantile(xs, .95)
        metrics[field+'_total'] = sum(xs)
    return dict(status='complete' if error is None else 'failed', error=error,
                completed=complete, request_count=len(reqs), clocks=clocks,
                requests=reqs, metrics=metrics, elapsed_ms=elapsed, rounds=rounds)


def save_episode(engine, scheduler, job, reqs):
    path = OUT/'runs'/(job['key']+'.json')
    if path.exists():
        saved = read(path)
        assert saved['job'] == job and saved['requests'] == reqs
        audit_result(saved, allow_timeout=job['phase']=='tune')
        return saved
    result = episode(engine, scheduler, reqs, job['method'], job['config'])
    trace = OUT/'rounds'/(job['key']+'.json.gz')
    trace.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(trace, 'wt') as f:
        json.dump(result.pop('rounds'), f)
    result.update(job=job, rounds_path=str(trace.relative_to(OUT)), rounds_sha256=digest(trace))
    write(path, result)
    print(job['key'], result['status'], result['metrics']['mean_ttft_ms'], flush=True)
    if result['status'] != 'complete':
        if job['phase']=='tune' and is_tuning_timeout(result):
            raise TuningTimeout('Recorded timeout: '+job['key'])
        raise RuntimeError(result['error'])
    return result


def check_smoke():
    assert read(OUT/'smoke.json') == dict(complete=True, incremental_kv_verified=True)
    for name in ('smoke_chunk64', 'smoke_chunk256', 'smoke_budget'):
        r = read(OUT/'runs'/(name+'.json'))
        audit_result(r)
        if name.startswith('smoke_chunk'):
            chunk = int(name.removeprefix('smoke_chunk'))
            with gzip.open(OUT/r['rounds_path'], 'rt') as f:
                first = json.load(f)[0]['allocations'][0]
            assert first['tokens']==chunk and first['blocks_after']==chunk//16


def is_tuning_timeout(r):
    return (r['job']['phase']=='tune' and r['status']=='failed'
            and r['error']=='TimeoutError: Predeclared episode deadline exceeded'
            and r['elapsed_ms']>=300000
            and 0<=r['completed']<r['request_count'])


def audit_result(r, allow_timeout=False):
    timeout = allow_timeout and is_tuning_timeout(r)
    assert (r['status']=='complete' or timeout)
    assert r['request_count']==len(r['requests'])
    assert digest(OUT/r['rounds_path'])==r['rounds_sha256']
    ttft = []
    finished = 0
    for request in r['requests']:
        c = r['clocks'][request['id']]
        assert 0<=c['tokens']<=request['output'] and c['tokens']==len(c['token_times_ms'])
        assert c['arrival_ms']==request['arrival_ms']
        assert c['first_ms']==(c['token_times_ms'][0] if c['tokens'] else None)
        if c['finish_ms'] is not None:
            assert c['tokens']==request['output'] and c['finish_ms']>=c['token_times_ms'][-1]
            finished += 1
        if c['tokens']:
            assert c['first_ms']>=c['arrival_ms']
            ttft.append(c['first_ms']-c['arrival_ms'])
        assert all(b>=a for a,b in zip(c['token_times_ms'], c['token_times_ms'][1:]))
    assert finished==r['completed']
    if timeout:
        assert r['metrics']['mean_ttft_ms'] is None and r['metrics']['p95_ttft_ms'] is None
        assert math.isclose(r['metrics']['completion_rate'], finished/len(r['requests']))
    else:
        assert finished==len(r['requests'])
        assert math.isclose(st.fmean(ttft), r['metrics']['mean_ttft_ms'], abs_tol=1e-6)
    with gzip.open(OUT/r['rounds_path'], 'rt') as f:
        rounds = json.load(f)
    assert len(rounds)==r['metrics']['round_count'] and rounds
    for i, row in enumerate(rounds):
        assert row['round']==i and 0<=row['free_after']<=row['total_blocks']
        assert set(row['scheduled']) <= set(r['clocks'])
        assert sum(row['scheduled'].values())<=row['budget']
        assert row['schedule_ms']+1e-6>=row['decision_ms']+row['native_ms']
        assert row['step_ms']+1e-6>=row['schedule_ms']
    for field, source in [('preemptions', 'preemptions'), ('recomputed_tokens', 'recomputed_tokens'),
                          ('interventions', 'intervened'), ('verified_rounds', 'verified'),
                          ('candidates', 'candidates')]:
        assert sum(row[source] for row in rounds)==r['metrics'][field]


def phase_run(manifest, phase, regime):
    engine, scheduler = build_engine('smoke' if phase=='smoke' else regime)
    # Fixed warmup before every fresh engine, excluded from measured episodes.
    warm = episode(engine, scheduler,
        [dict(id=f'warm{i}', prompt=256, output=16, arrival_ms=0) for i in range(4)],
        'srp', CONFIGS[1])
    assert warm['status']=='complete', warm['error']
    if phase=='smoke':
        for chunk in (64, 256):
            job = dict(key=f'smoke_chunk{chunk}', phase='smoke', regime='smoke',
                       method='native', arm='native', config=dict(budget=256, chunk=chunk, concurrency=16))
            saved = save_episode(engine, scheduler, job,
                     [dict(id='incremental', prompt=1024, output=8, arrival_ms=0)])
            with gzip.open(OUT/saved['rounds_path'], 'rt') as f:
                first = json.load(f)[0]['allocations'][0]
            assert first['tokens']==chunk and first['blocks_after']==chunk//16, first
        job = dict(key='smoke_budget', phase='smoke', regime='smoke', method='budget',
                   arm='budget', config=CONFIGS[2])
        save_episode(engine, scheduler, job,
                     [dict(id=f'budget{i}', prompt=4096, output=8, arrival_ms=0) for i in range(2)])
        write(OUT/'smoke.json', dict(complete=True, incremental_kv_verified=True))
        return
    selection = read(OUT/'selection.json') if phase=='compare' else None
    for job in jobs(manifest, phase, regime, selection):
        save_episode(engine, scheduler, job, requests_for(manifest['blocks'][job['block']], regime))


def load_expected(manifest, phase, selection=None):
    result = []
    for regime in REGIMES:
        for job in jobs(manifest, phase, regime, selection):
            path = OUT/'runs'/(job['key']+'.json')
            if not path.exists():
                raise RuntimeError('Missing run: '+job['key'])
            r = read(path)
            assert r['job']==job and r['requests']==requests_for(manifest['blocks'][job['block']], regime)
            audit_result(r, allow_timeout=phase=='tune')
            result.append(r)
    return result


def check_aa(manifest):
    rows = load_expected(manifest, 'aa')
    groups = {}
    for r in rows:
        key = r['job']['regime']+'/'+r['job']['block']
        groups.setdefault(key, []).append(r['metrics']['mean_ttft_ms'])
    result = {k: dict(ttft_ms=v, range_over_mean=(max(v)-min(v))/st.fmean(v)) for k,v in groups.items()}
    passed = all(v['range_over_mean'] <= manifest['aa_max_range_over_mean'] for v in result.values())
    write(OUT/'aa_check.json', dict(passed=passed, groups=result))
    if not passed:
        raise RuntimeError('A/A variation exceeds predeclared 20% range/mean gate; review aa_check.json')


def select(manifest):
    rows = load_expected(manifest, 'tune')
    selected = {}
    reports = {}
    for regime in REGIMES:
        group = [r for r in rows if r['job']['regime']==regime]
        bases = {}
        for block in {r['job']['block'] for r in group}:
            # A timed-out SRP reference is never silently dropped or imputed.
            # Prefer the fixed middle configuration, then lowest config index.
            refs = [r for cfg in (CONFIGS[1], CONFIGS[0], CONFIGS[2]) for r in group
                    if r['job']['block']==block and r['job']['arm']=='srp'
                    and r['job']['config']==cfg and r['status']=='complete']
            if not refs:
                raise RuntimeError(f'No complete SRP normalization reference: {regime}/{block}')
            bases[block] = refs[0]['metrics']['mean_ttft_ms']
        selected[regime] = {}
        reports[regime] = {}
        for method in METHODS:
            scores = []
            for cfg in CONFIGS:
                pair = [r for r in group if r['job']['arm']==method and r['job']['config']==cfg]
                assert len(pair)==len(bases)
                scores.append(st.fmean(r['metrics']['mean_ttft_ms']/bases[r['job']['block']]
                                       for r in pair) if all(r['status']=='complete' for r in pair) else None)
            reports[regime][method] = scores
            eligible = [i for i,s in enumerate(scores) if s is not None]
            if not eligible:
                write(OUT/'tuning_report.json', dict(scores=reports,
                      timeouts=[r['job']['key'] for r in rows if r['status']=='failed']))
                raise RuntimeError(f'No configuration completed both blocks: {regime}/{method}')
            selected[regime][method] = CONFIGS[min(eligible, key=lambda i:(scores[i], i))]
    write(OUT/'tuning_report.json', dict(scores=reports,
          timeouts=[r['job']['key'] for r in rows if r['status']=='failed']))
    target = OUT/'selection.json'
    if target.exists():
        assert read(target)==selected
    else:
        write(target, selected)
    return selected


def summarize(manifest):
    check_smoke()
    check_aa(manifest)
    selection = select(manifest)
    tuning_failures = [r for r in load_expected(manifest, 'tune') if r['status']=='failed']
    rows = load_expected(manifest, 'compare', selection)
    metrics, comparisons = [], {}
    for regime in REGIMES:
        group = [r for r in rows if r['job']['regime']==regime]
        for arm in ARMS:
            subset = [r for r in group if r['job']['arm']==arm]
            metrics.append(dict(regime=regime, arm=arm,
                **{k: st.fmean(r['metrics'][k] for r in subset) for k in subset[0]['metrics']
                   if all(r['metrics'][k] is not None for r in subset)}))
        comparisons[regime] = {}
        for arm in ('native', 'srp', 'srp_matched', 'shadow'):
            pairs = []
            for block in sorted({r['job']['block'] for r in group}):
                a = st.fmean(r['metrics']['mean_ttft_ms'] for r in group if r['job']['block']==block and r['job']['arm']=='budget')
                b = st.fmean(r['metrics']['mean_ttft_ms'] for r in group if r['job']['block']==block and r['job']['arm']==arm)
                pairs.append(dict(block=block, budget_ttft_ms=a, baseline_ttft_ms=b,
                                  gain_ms=b-a, loss_fraction=a/b-1))
            rng = random.Random(20260922)
            samples = [st.fmean(rng.choice(pairs)['gain_ms'] for _ in pairs) for _ in range(5000)]
            comparisons[regime][arm] = dict(mean_gain_ms=st.fmean(p['gain_ms'] for p in pairs),
                bootstrap95_ms=[quantile(samples, .025), quantile(samples, .975)],
                positive_blocks=sum(p['gain_ms']>0 for p in pairs), pairs=pairs)
    write(OUT/'summary.json', dict(complete=True, expected=dict(smoke=3, aa=18, tune=54, compare=180),
          failed_runs=[dict(key=r['job']['key'], error=r['error'],
                   completed=r['completed'], requests=r['request_count'],
                   preemptions=r['metrics']['preemptions'],
                   recomputed_tokens=r['metrics']['recomputed_tokens']) for r in tuning_failures],
          all_attempts_accounted=True, all_requests_completed=not tuning_failures,
          audit_errors=[], comparisons=comparisons, metrics=metrics,
          claim=manifest['claim'], implementation=manifest['implementation'],
          hardware=manifest['hardware'],
          cost_note='All online decision, native scheduling and common audit costs remain in TTFT/TBT/throughput; no zero-cost subtraction'))
    print('Completed audited matrix:', OUT/'summary.json', flush=True)


def import_prior(manifest):
    source = os.environ.get('V1_BUDGET_IMPORT')
    if not source:
        return
    source = Path(source).resolve()
    if source==OUT or not source.is_dir():
        raise RuntimeError('Import source must be an existing, separate result directory')
    old = read(source/'manifest.json')
    assert old['schema']==1 and old['implementation']=='V1-budget-light-1'
    for key in ('hardware', 'model', 'versions', 'model_revision', 'blocks',
                'configs', 'regimes', 'methods', 'arms', 'repeats', 'restricted_blocks',
                'timeout_s', 'aa_max_range_over_mean', 'claim', 'engine'):
        if old[key] != manifest[key]:
            raise RuntimeError('Previous matrix mismatch: '+key)
    # Runner, tests and protocol changed. Native policy, scheduler, engine,
    # inputs, model and dependencies must be identical before reusing results.
    changeable = ('/run.py', '/test_runner.py', '/bootstrap.sh', '/PROTOCOL.md')
    for key, value in old['source_hashes'].items():
        if key not in manifest['source_hashes'] or (
                not key.endswith(changeable) and manifest['source_hashes'][key]!=value):
            raise RuntimeError('Previous source mismatch: '+key)
    assert read(source/'capacity.json')['blocks']>768
    permitted = {j['key']:j for phase in ('aa','tune') for regime in REGIMES
                 for j in jobs(manifest, phase, regime)}
    imported = []
    for path in sorted((source/'runs').glob('*.json')):
        row = read(path)
        job = row['job']
        if job['phase']=='compare':
            raise RuntimeError('Do not import comparisons from a different scoring protocol')
        if job['phase']=='smoke':
            assert job['key'] in ('smoke_chunk64', 'smoke_chunk256', 'smoke_budget')
        else:
            assert permitted[job['key']]==job
            assert row['requests']==requests_for(manifest['blocks'][job['block']], job['regime'])
        assert path.name==job['key']+'.json'
        prior_trace = source/row['rounds_path']
        assert prior_trace.is_file() and digest(prior_trace)==row['rounds_sha256']
        local_trace = OUT/row['rounds_path']
        local_trace.parent.mkdir(parents=True, exist_ok=True)
        if not local_trace.exists():
            shutil.copy2(prior_trace, local_trace)
        local_path = OUT/'runs'/path.name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            shutil.copy2(path, local_path)
        if digest(local_path)!=digest(path):
            raise RuntimeError('Imported run conflicts with existing record: '+job['key'])
        audit_result(read(local_path), allow_timeout=job['phase']=='tune')
        imported.append(job['key'])
    for name in ('capacity.json', 'smoke.json'):
        prior = source/name
        if prior.exists():
            target = OUT/name
            if not target.exists():
                shutil.copy2(prior, target)
            assert digest(prior)==digest(target)
    if (OUT/'smoke.json').exists():
        check_smoke()
    provenance = dict(source=str(source), manifest_sha256=digest(source/'manifest.json'),
                      imported=imported, timeout_runs=[key for key in imported
                          if read(OUT/'runs'/(key+'.json'))['status']=='failed'])
    target = OUT/'migration.json'
    if target.exists():
        assert read(target)==provenance
    else:
        write(target, provenance)
    print('Verified previous results:', len(imported),
          'record(s); timeout(s):', len(provenance['timeout_runs']), flush=True)


def orchestrate():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = initialize()
        import_prior(manifest)
        used = int(subprocess.check_output(['nvidia-smi', '--id='+manifest['hardware']['gpu'],
                   '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if used > 1024:
            raise RuntimeError(f'Target GPU is occupied ({used} MiB); no processes were stopped')
        logs = OUT/'logs'
        logs.mkdir(exist_ok=True)
        for phase in ('smoke', 'aa', 'tune', 'compare'):
            for regime in (('smoke',) if phase=='smoke' else REGIMES):
                if phase=='smoke' and (OUT/'smoke.json').exists():
                    check_smoke()
                    continue
                selection = read(OUT/'selection.json') if phase=='compare' else None
                todo = jobs(manifest, phase, regime, selection) if phase!='smoke' else []
                label = phase+'_'+regime
                for attempt in range(len(todo)+1):
                    if todo and all((OUT/'runs'/(j['key']+'.json')).exists() for j in todo):
                        break
                    print('Running', label, 'attempt', attempt+1, flush=True)
                    with (logs/(label+'.log')).open('a') as log:
                        process = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                            '--phase', phase, '--regime', regime], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, timeout=6*3600)
                    if process.returncode!=0 and not (phase=='tune' and process.returncode==75):
                        raise RuntimeError(f'{label} exited {process.returncode}; see {logs/(label+".log")}')
                    if phase=='smoke' and process.returncode==0:
                        check_smoke()
                        break
                else:
                    raise RuntimeError('Too many attempts: '+label)
                if phase!='smoke':
                    for job in todo:
                        row = read(OUT/'runs'/(job['key']+'.json'))
                        assert row['job']==job
                        audit_result(row, allow_timeout=phase=='tune')
            if phase=='aa':
                check_aa(manifest)
            if phase=='tune':
                select(manifest)
        summarize(manifest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('all', 'smoke', 'aa', 'tune', 'compare', 'summary'), default='all')
    p.add_argument('--regime', choices=(*REGIMES, 'smoke'), default='queue')
    a = p.parse_args()
    if os.environ.get('VLLM_USE_V1')!='1' or os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')!='0':
        raise RuntimeError('Use bootstrap.sh: V1 and synchronous in-process engine required')
    try:
        if a.phase=='all':
            orchestrate()
        else:
            manifest = initialize()
            if a.phase=='summary':
                summarize(manifest)
            else:
                if a.phase in ('tune', 'compare'):
                    check_aa(manifest)
                if a.phase=='compare':
                    select(manifest)
                phase_run(manifest, a.phase, a.regime)
    except TuningTimeout as exc:
        print(exc, flush=True)
        return 75
    except Exception as exc:
        write(OUT/'last_error.json', dict(phase=a.phase, regime=a.regime,
                                        error=f'{type(exc).__name__}: {exc}'))
        raise
    return 0


if __name__=='__main__':
    sys.exit(main())
