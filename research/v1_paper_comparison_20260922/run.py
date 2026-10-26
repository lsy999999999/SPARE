"""Frozen first comparison; all GPU work occurs in restartable phase workers."""
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
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.v1_budget_20260922 import run as ref

HERE = Path(__file__).resolve().parent
OLD = Path(os.environ.get('PAPER_COMPARE_REFERENCE', ROOT/'artifacts/v1_budget_gpu2_20260922_r2'))
OUT = Path(os.environ.get('PAPER_COMPARE_OUTPUT', ROOT/'artifacts/v1_paper_comparison_gpu2_20260922_r2'))
ref.OUT = OUT
PAPER = ('sarathi_v1', 'past_future_v1')
ARMS = ('budget', 'srp') + PAPER
REGIMES = ref.REGIMES
CONFIGS = [dict(budget=b, chunk=b, concurrency=c) for b,c in ((512,16),(1024,32),(2048,64))]
read, write, digest = ref.read, ref.write, ref.digest


def history_data(manifest):
    dm = read(ref.DATA/'data_manifest.json')
    histories = {}
    for source in ('code', 'conversation'):
        spec = dm['sources'][source]['splits']['train']
        path = ROOT/spec['path']
        assert digest(path) == spec['sha256']
        excluded = {r['request_id'] for b in manifest['blocks'].values()
                    if b['role']=='dev' and b['source']==source for r in b['rows']}
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        chosen = [r for r in rows if r['request_id'] not in excluded][-1000:]
        assert len(chosen) == 1000
        histories[source] = dict(source_sha256=spec['sha256'],
                                 request_ids=[r['request_id'] for r in chosen],
                                 lengths=[r['actual_output_tokens'] for r in chosen])
    return histories


def initialize():
    previous = read(OLD/'manifest.json')
    assert read(OLD/'summary.json')['complete']
    assert ref.hardware() == previous['hardware']
    assert str(ref.MODEL) == previous['model']
    for name in ('policy.py', 'scheduler.py', 'run.py'):
        assert digest(ROOT/'research/v1_budget_20260922'/name) == previous['source_hashes']['research/v1_budget_20260922/'+name]
    for package, version in previous['versions'].items():
        assert importlib.metadata.version(package) == version
    dist = importlib.metadata.distribution('vllm')
    for name, sha in previous['source_hashes'].items():
        if name.startswith('vllm/'):
            assert digest(dist.locate_file(name)) == sha
    histories = history_data(previous)
    provenance = {name:digest(OLD/name) for name in ('manifest.json','selection.json',
                  'summary.json','capacity.json','tuning_report.json')}
    tuning_hashes = {p.name:digest(p) for p in sorted((OLD/'runs').glob('tune_*.json'))}
    manifest = dict(schema=1, implementation='a-paper-rules-v1/cpu-affinity-r2',
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        hardware=previous['hardware'], model=previous['model'], versions=previous['versions'],
        model_revision=previous['model_revision'], blocks=previous['blocks'],
        reference=str(OLD), reference_hashes=provenance, archived_tuning_hashes=tuning_hashes,
        inherited_selections=read(OLD/'selection.json'), paper_configs=CONFIGS,
        histories=histories, arms=list(ARMS), regimes=list(REGIMES), repeats=2,
        timeout_s=300, capacity=read(OLD/'capacity.json')['blocks'],
        source_hashes={p.name:digest(p) for p in sorted(HERE.iterdir()) if p.is_file()},
        frozen_reference_hashes=previous['source_hashes'],
        claim='Exploratory same-source comparison of frozen A and V1 paper-rule adaptations')
    path = OUT/'manifest.json'
    if path.exists():
        assert read(path)==manifest, 'Manifest changed; use a new output directory'
    else:
        write(path, manifest)
        for p in HERE.iterdir():
            if p.is_file():
                dest = OUT/'source'/p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(p.read_bytes())
        write(OUT/'capacity.json', dict(blocks=manifest['capacity']))
    return manifest


def jobs(m, phase, regime, selection=None):
    if phase=='smoke':
        return [dict(key='smoke_'+method, phase=phase, regime='restricted', block=None,
                     arm=method, method=method, config=CONFIGS[1], repeat=0) for method in PAPER]
    role = 'compare' if phase=='compare' else 'dev'
    blocks = sorted(k for k,b in m['blocks'].items() if b['role']==role)
    result = []
    for rep in range(2 if phase=='compare' else 3 if phase=='aa' else 1):
        for bi, block in enumerate(blocks):
            arms = list(ARMS if phase=='compare' else PAPER if phase=='tune' else ('srp',))
            shift = (bi+rep) % len(arms)
            arms = arms[shift:]+arms[:shift]
            if rep % 2:
                arms.reverse()
            for ci in range(3 if phase=='tune' else 1):
                for arm in arms:
                    cfg = (CONFIGS[ci] if phase=='tune' else ref.CONFIGS[1])
                    if phase=='compare':
                        cfg = selection[regime][arm]
                    result.append(dict(key=f'{phase}_{regime}_{block}_{arm}_c{ci}_r{rep}',
                        phase=phase, regime=regime, block=block, arm=arm, method=arm,
                        config=cfg, repeat=rep))
    return result


def requests(m, job):
    if job['phase']=='smoke':
        return [dict(id=f'smoke{i}', prompt=512+128*i, output=32, arrival_ms=0) for i in range(6)]
    return ref.requests_for(m['blocks'][job['block']], job['regime'])


def audit(row):
    if row['status']=='complete':
        ref.audit_result(row)
    else:
        assert row['error']=='TimeoutError: Predeclared episode deadline exceeded', row['error']
        clone = dict(row, job=dict(row['job'], phase='tune'))
        ref.audit_result(clone, allow_timeout=True)


def build_engine(regime, m):
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    expected = m['hardware']['uuid'].removeprefix('GPU-').lower()
    assert str(torch.cuda.get_device_properties(0).uuid).lower() == expected
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=str(ref.MODEL), dtype='half', max_model_len=12288,
        max_num_seqs=64, max_num_batched_tokens=2048,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        block_size=16, gpu_memory_utilization=.9, enforce_eager=True,
        disable_log_stats=True, seed=0,
        num_gpu_blocks_override=768 if regime=='restricted' else m['capacity'],
        scheduler_cls='research.v1_paper_comparison_20260922.scheduler.ComparisonScheduler'))
    scheduler = engine.engine_core.engine_core.scheduler
    assert scheduler.kv_cache_config.num_blocks==(768 if regime=='restricted' else m['capacity'])
    warm = ref.episode(engine, scheduler,
        [dict(id=f'warm{i}', prompt=256, output=16, arrival_ms=0) for i in range(4)],
        'srp', ref.CONFIGS[1])
    assert warm['status']=='complete', warm['error']
    return engine, scheduler


def phase_run(m, phase, regime):
    selection = read(OUT/'selection.json') if phase=='compare' else None
    todo = []
    for job in jobs(m, phase, regime, selection):
        path = OUT/'runs'/(job['key']+'.json')
        if path.exists():
            row = read(path)
            assert row['job']==job and row['requests']==requests(m, job)
            audit(row)
        else:
            todo.append(job)
    if not todo:
        return
    engine, scheduler = build_engine('restricted' if phase=='smoke' else regime, m)
    for job in todo:
        history = ([32]*1000 if phase=='smoke' else
            m['histories'][m['blocks'][job['block']]['source']]['lengths'])
        scale = 1 if regime=='queue' or phase=='smoke' else 4
        scheduler.history_context = [n*scale for n in history]
        scheduler.seed_context = 20260922+job['repeat']
        result = ref.episode(engine, scheduler, requests(m, job), job['method'], job['config'])
        rounds = result.pop('rounds')
        for key in ('gate_checked','gate_rejected'):
            result['metrics'][key] = sum(r.get(key,0) for r in rounds)
        trace = OUT/'rounds'/(job['key']+'.json.gz')
        trace.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(trace, 'wt') as f:
            json.dump(rounds, f)
        result.update(job=job, rounds_path=str(trace.relative_to(OUT)), rounds_sha256=digest(trace))
        write(OUT/'runs'/(job['key']+'.json'), result)
        print(job['key'], result['status'], result['metrics']['mean_ttft_ms'], flush=True)
        audit(result)
        if result['status']!='complete':
            if phase in ('smoke','aa'):
                raise RuntimeError('Initial verification timed out')
            raise SystemExit(75)


def load_phase(m, phase, selection=None):
    result = []
    for regime in (('restricted',) if phase=='smoke' else REGIMES):
        for job in jobs(m,phase,regime,selection):
            row = read(OUT/'runs'/(job['key']+'.json'))
            assert row['job']==job and row['requests']==requests(m,job)
            audit(row)
            result.append(row)
    return result


def check_aa(m):
    rows = load_phase(m,'aa')
    groups = {}
    for row in rows:
        key = row['job']['regime']+'/'+row['job']['block']
        if 'gpu' in row['job']:
            key += '/gpu'+str(row['job']['gpu'])
        groups.setdefault(key,[]).append(row['metrics']['mean_ttft_ms'])
    result = {k:dict(ttft_ms=v,range_over_mean=(max(v)-min(v))/st.fmean(v)) for k,v in groups.items()}
    passed = all(v['range_over_mean']<=.20 for v in result.values())
    write(OUT/'aa_check.json',dict(passed=passed,groups=result))
    assert passed, 'A/A noise exceeds frozen threshold'


def select(m):
    rows = load_phase(m,'tune')
    selected, scores = {}, {}
    for regime in REGIMES:
        selected[regime] = {a:m['inherited_selections'][regime][a] for a in ('budget','srp')}
        scores[regime] = {}
        for arm in PAPER:
            values = []
            for cfg in CONFIGS:
                pairs = [r for r in rows if r['job']['regime']==regime
                         and r['job']['arm']==arm and r['job']['config']==cfg]
                assert len(pairs)==2
                ratios = []
                for row in pairs:
                    path = OLD/'runs'/f"tune_{regime}_{row['job']['block']}_srp_c1_r0.json"
                    assert digest(path)==m['archived_tuning_hashes'][path.name]
                    base = read(path)
                    assert base['status']=='complete'
                    if row['status']=='complete':
                        ratios.append(row['metrics']['mean_ttft_ms']/base['metrics']['mean_ttft_ms'])
                values.append(st.fmean(ratios) if len(ratios)==2 else None)
            scores[regime][arm] = values
            eligible = [i for i,v in enumerate(values) if v is not None]
            write(OUT/'tuning_report.json',dict(scores=scores,
                failures=[r['job']['key'] for r in rows if r['status']!='complete']))
            if not eligible:
                raise RuntimeError(f'No complete configuration: {regime}/{arm}')
            selected[regime][arm] = CONFIGS[min(eligible,key=lambda i:(values[i],i))]
    if (OUT/'selection.json').exists():
        assert read(OUT/'selection.json')==selected
    write(OUT/'selection.json',selected)
    return selected


def summarize(m):
    check_aa(m)
    selection = select(m)
    smoke = load_phase(m,'smoke')
    rows = load_phase(m,'compare',selection)
    tuning = load_phase(m,'tune')
    metrics, comparisons = [], {}
    for regime in REGIMES:
        group = [r for r in rows if r['job']['regime']==regime]
        for arm in ARMS:
            subset = [r for r in group if r['job']['arm']==arm]
            complete = all(r['status']=='complete' for r in subset)
            metric = dict(regime=regime,arm=arm,complete=complete,runs=len(subset),
                          completed=sum(r['completed'] for r in subset),
                          requests=sum(r['request_count'] for r in subset))
            if complete:
                metric.update({k:st.fmean(r['metrics'][k] for r in subset)
                    for k in subset[0]['metrics'] if all(r['metrics'][k] is not None for r in subset)})
            metrics.append(metric)
        comparisons[regime] = {}
        for arm in ('srp',)+PAPER:
            pairs = []
            for block in sorted({r['job']['block'] for r in group}):
                a = [r for r in group if r['job']['block']==block and r['job']['arm']=='budget']
                b = [r for r in group if r['job']['block']==block and r['job']['arm']==arm]
                if not all(r['status']=='complete' for r in a+b):
                    pairs.append(dict(block=block,complete=False))
                    continue
                av = st.fmean(r['metrics']['mean_ttft_ms'] for r in a)
                bv = st.fmean(r['metrics']['mean_ttft_ms'] for r in b)
                pairs.append(dict(block=block,complete=True,a_ttft_ms=av,baseline_ttft_ms=bv,
                                  gain_ms=bv-av,gain_fraction=(bv-av)/bv))
            value = dict(pairs=pairs,complete=all(p['complete'] for p in pairs))
            if value['complete']:
                rng = random.Random(20260922)
                draws = [st.fmean(rng.choice(pairs)['gain_ms'] for _ in pairs) for _ in range(5000)]
                value.update(mean_gain_ms=st.fmean(p['gain_ms'] for p in pairs),
                    positive_blocks=sum(p['gain_ms']>0 for p in pairs),
                    bootstrap95_ms=[ref.quantile(draws,.025),ref.quantile(draws,.975)])
            comparisons[regime][arm] = value
    failures = [dict(key=r['job']['key'],error=r['error'],completed=r['completed'],
                     requests=r['request_count']) for r in tuning+rows if r['status']!='complete']
    summary = dict(complete=True,all_attempts_accounted=True,
        all_requests_completed=not failures,expected=m.get('expected',dict(smoke=2,aa=18,tune=36,compare=96)),
        failed_runs=failures,metrics=metrics,comparisons=comparisons,
        selection=selection,hardware=m['hardware'],claim=m['claim'],audit_errors=[])
    write(OUT/'summary.json',summary)
    lines = ['# Current A versus V1 paper-rule adaptations','',m['claim'],'',
        '| Regime | Method | Mean TTFT (s) | Output token/s | Preemptions/run | Completed |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in metrics:
        if row['complete']:
            lines.append(f"| {row['regime']} | {row['arm']} | {row['mean_ttft_ms']/1000:.3f} | {row['throughput_tokens_s']:.2f} | {row['preemptions']:.2f} | {row['completed']}/{row['requests']} |")
        else:
            lines.append(f"| {row['regime']} | {row['arm']} | incomplete | — | — | {row['completed']}/{row['requests']} |")
    lines += ['', 'A/SRP policy unchanged; all methods within each comparison block run on the same GPU.',
              'Paper names denote the exact rule adaptations documented in source/PROTOCOL.md.',
              'All online decision and diagnostic costs remain in measured latency/throughput.',
              'Each regime has four source blocks and two repetitions per method.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('Completed audited matrix',OUT,flush=True)


def orchestrate(m):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for phase in ('smoke','aa','tune','compare'):
            selection = read(OUT/'selection.json') if phase=='compare' else None
            for regime in (('restricted',) if phase=='smoke' else REGIMES):
                planned = jobs(m,phase,regime,selection)
                attempts = 0
                while any(not (OUT/'runs'/(j['key']+'.json')).exists() for j in planned):
                    used = int(subprocess.check_output(['nvidia-smi','--id='+m['hardware']['gpu'],
                        '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                    if used>1024:
                        raise RuntimeError(f'Target GPU is occupied: {used} MiB')
                    log = OUT/'logs'/f'{phase}_{regime}.log'
                    log.parent.mkdir(parents=True,exist_ok=True)
                    with log.open('a') as f:
                        code = subprocess.run([sys.executable,'-B','-u',str(HERE/'run.py'),
                            '--phase',phase,'--regime',regime],stdout=f,stderr=subprocess.STDOUT,
                            timeout=21600).returncode
                    if code not in (0,75):
                        raise RuntimeError(f'{phase}/{regime} exited {code}; see {log}')
                    attempts += 1
                    if attempts>len(planned)+1:
                        raise RuntimeError('No forward progress')
            if phase=='aa':
                check_aa(m)
            if phase=='tune':
                select(m)
        summarize(m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',choices=('all','smoke','aa','tune','compare','summary'),default='all')
    p.add_argument('--regime',choices=REGIMES,default='restricted')
    args = p.parse_args()
    assert os.environ.get('VLLM_USE_V1')=='1'
    assert os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')=='0'
    m = initialize()
    if args.phase=='all':
        orchestrate(m)
    elif args.phase=='summary':
        summarize(m)
    else:
        phase_run(m,args.phase,args.regime)


if __name__=='__main__':
    main()
