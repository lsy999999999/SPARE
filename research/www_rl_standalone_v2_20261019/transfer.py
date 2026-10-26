"""Frozen-policy KV sweeps and paired Code/Conversation mixture experiments."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from research.www_rl_standalone_v2_20261019 import frontier as prior

ROOT = prior.ROOT
OUT = ROOT/'artifacts/www_rl7047_transfer_20261020_r1'
BASE = prior.train.run.old.base
ORIGINAL_GENERATE = BASE.generate
PROPORTIONS = (0,25,50,75,100)
REFERENCE_QPS = (2.,3.)
COARSE = (.25,.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,7.,8.,10.,12.,16.)


def mixed_stream(sample, rate):
    """Nested balanced four-arrival blocks; paired lengths; common arrivals."""
    original = ORIGINAL_GENERATE(sample,rate)
    code = BASE.read(ROOT/sample['code_input'])
    assert len(code)==64
    seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:8],'big')
    picks = np.random.default_rng(seed ^ 0x9E3779B97F4A7C15).integers(0,64,size=len(original))
    ranks = np.random.default_rng(seed ^ 0xA70120261020).random((1024,4)).argsort(axis=1).reshape(-1)
    count_code = sample['code_pct']//25
    labels = ranks[:len(original)] < count_code
    result = []
    for i,row in enumerate(original):
        replacement = code[int(picks[i])]
        result.append(dict(row,prompt=replacement['prompt'],output=replacement['output'])
                      if labels[i] else row)
    assert [r['arrival_ms'] for r in result]==[r['arrival_ms'] for r in original]
    if sample['code_pct']==0:
        assert result==original
    return result,labels.tolist()


def stream_info(sample,rate):
    requests,labels=mixed_stream(sample,rate)
    measured=[i for i,r in enumerate(requests) if 20000<=r['arrival_ms']<60000]
    return dict(stream_sha256=hashlib.sha256(json.dumps(requests,sort_keys=True).encode()).hexdigest(),
        total_requests=len(requests),code_requests=sum(labels),
        measured_stream_requests=len(measured),measured_code_requests=sum(labels[i] for i in measured))


def worker(sample,kv,arm,rate):
    if 'code_pct' not in sample:
        return prior.window_job(sample,kv,arm,rate)
    def generate(s,r):
        return mixed_stream(s,r)[0]
    with patch.object(BASE,'generate',generate):
        row=prior.window_job(sample,kv,arm,rate)
    row.update(stream_info(sample,rate),code_pct=sample['code_pct'])
    return row


def passes(s,rate):
    return (s['failed_samples']==0 and s['mean_ttft_ms']<=2000
            and s['p99_tbt_ms']<=200 and s['backlog_drift']<=max(2,rate))


def signature(cohort):
    data=[dict(name=s['name'],source=s['source'],input_sha256=prior.digest(ROOT/s['input']),
               code_pct=s.get('code_pct'),
               code_input_sha256=prior.digest(ROOT/s['code_input']) if 'code_input' in s else None)
          for s in cohort]
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def scan(task,pool,selected):
    stage,source,kv,arm,pct=task
    cohort=prior.sweep.cohort_for(source)
    if stage=='mix':
        codes={s['name'].replace('code_',''):s for s in prior.sweep.cohort_for('code')}
        cohort=[dict(s,code_pct=pct,code_input=codes[s['name'].replace('conversation_','')]['input'])
                for s in cohort]
    name=(f'mix_p{pct:03d}_c{kv}_{arm}' if stage=='mix' else f'kv_{source}_c{kv}_{arm}')
    target=OUT/'groups'/f'{name}.json'
    sig=signature(cohort)
    sha=selected['sha256'] if arm!='spare' else None
    bank={}
    if stage=='kv' or pct==0:
        for p in (prior.OUT/'rates').glob(f'{source}_c{kv}_{arm}_seed7047_r*.json'):
            bank[float(p.stem.split('_r')[-1])]=p
    # Full-SPARE 1024 records share the same archived workload and clock.
    if stage=='kv' and arm=='spare' and kv==1024:
        d=prior.read(prior.ABLATION/'design.json')
        assert d['timing_sha256']==prior.digest(BASE.PREVIOUS/'timing_profile.json')
        assert {(s['name'],s['input']) for s in cohort}=={
            (s['name'],s['input']) for s in d['samples'] if s['source']==source}
        for p in (prior.ABLATION/'rates').glob(f'{source}_c{kv}_spare_r*.json'):
            bank[float(p.stem.split('_r')[-1])]=p
    cache,files={},{}

    def evaluate(rate):
        rate=round(rate,2)
        if rate in cache:return cache[rate]
        path=OUT/'rates'/f'{name}_r{rate:g}.json'
        if path.exists():
            data=prior.read(path)
        elif rate in bank:
            old=bank[rate]
            rows=prior.read(old)['rows']
            if stage=='mix':
                mapping={s['name']:s for s in cohort}
                rows=[dict(r,**stream_info(mapping[r['name']],rate),code_pct=pct) for r in rows]
            data=dict(rows=rows,reused_from=str(old.relative_to(ROOT)),reused_sha256=prior.digest(old),
                      input_sha256=sig,checkpoint_sha256=sha)
            data['summary']=prior.train.run.old.prior.old.summary(rows,rate)
            prior.write(path,data)
        else:
            jobs=[pool.submit(worker,s,kv,arm,rate) for s in cohort]
            rows=sorted((f.result() for f in jobs),key=lambda r:r['name'])
            data=dict(rows=rows,summary=prior.train.run.old.prior.old.summary(rows,rate),
                      newly_simulated=50,input_sha256=sig,checkpoint_sha256=sha)
            prior.write(path,data)
            print(f'NEW {name} QPS={rate:g} TTFT={data["summary"]["mean_ttft_ms"]}',flush=True)
        assert data['input_sha256']==sig and data['checkpoint_sha256']==sha
        assert len(data['rows'])==50 and {r['name'] for r in data['rows']}=={s['name'] for s in cohort}
        assert data['summary']==prior.train.run.old.prior.old.summary(data['rows'],rate)
        cache[rate]=data['summary'];files[rate]=path
        return cache[rate]

    for rate in sorted(bank):evaluate(rate)
    for path in (OUT/'rates').glob(f'{name}_r*.json'):
        evaluate(float(path.stem.split('_r')[-1]))
    for rate in REFERENCE_QPS:evaluate(rate)
    # Same ascending coarse grid for all mixtures, even when only the endpoints
    # have legacy fine observations. All legacy records remain part of the audit.
    for rate in COARSE:
        if not passes(evaluate(rate),rate):break

    def bracket():
        low=None
        for r in sorted(cache):
            if not passes(cache[r],r):return low,r
            low=r
        raise RuntimeError('no failure through scanned rates')
    low,high=bracket()
    while low is not None and high-low>.0200001:
        mid=round((low+high)*50)/100
        evaluate(mid)
        low,high=bracket()
    fail=cache[high]
    reasons=[]
    if fail['failed_samples']:reasons.append('invalid_window')
    else:
        if fail['mean_ttft_ms']>2000:reasons.append('mean_ttft')
        if fail['p99_tbt_ms']>200:reasons.append('p99_tbt')
        if fail['backlog_drift']>max(2,high):reasons.append('backlog_drift')
    result=dict(stage='complete',experiment=stage,source=source,kv_blocks=kv,arm=arm,
        code_pct=pct,seed=7047,checkpoint_sha256=sha,input_sha256=sig,windows=50,
        mean_ttft_limit_ms=2000,capacity_qps=low,next_failed_qps=high,
        resolution_qps=round(high-low,4) if low is not None else None,
        first_failure_reasons=reasons,
        reference_metrics={str(r):dict(summary=cache[r],passed=passes(cache[r],r)) for r in REFERENCE_QPS},
        records=[dict(rate_qps=r,path=str(files[r].relative_to(ROOT)),sha256=prior.digest(files[r]),
                      summary=cache[r]) for r in sorted(cache)])
    prior.write(target,result)
    print(f'DONE {name} capacity={low} next={high}',flush=True)
    return result


def prepare(stage,kvs):
    selected=prior.read(prior.ORIGINAL/'checkpoints/seed7047_status.json')['selected']
    assert selected['episode']==256
    assert prior.digest(prior.ORIGINAL/selected['path'])==selected['sha256']
    tasks=([(stage,'conversation',2048,arm,pct) for pct in PROPORTIONS
            for arm in ('spare','rl_standalone')] if stage=='mix' else
           [(stage,source,kv,arm,None) for source in ('conversation','code') for kv in kvs
            for arm in ('spare','rl_standalone')])
    design=dict(stage=stage,kvs=kvs if stage=='kv' else [2048],code_percentages=PROPORTIONS,
        selected=selected,tasks=tasks,reference_qps=REFERENCE_QPS,mean_ttft_limit_ms=2000,
        windows=50,capacity_rule='ascending observed QPS through first failure; all valid; '
        'mean TTFT <=2s; mean window P99 TBT<=200ms; mean backlog drift<=max(2,QPS)',
        mixture_protocol='Common Conversation arrival times and indices into paired 64-request pools; '
        'nested random permutations in blocks of four arrivals set Code share; preserve prompt/output pairs. '
        'History fixed to Conversation for all proportions; new pure-Code endpoint uses common arrivals. '
        'Partial blocks cause small realized-share deviations, recorded per window.',
        timing_sha256=prior.digest(BASE.PREVIOUS/'timing_profile.json'),
        source_sha256=prior.digest(Path(__file__)))
    prior.write(OUT/f'design_{stage}.json',design)
    return tasks,selected


def smoke(pool,stage):
    sample=prior.sweep.cohort_for('conversation')[0]
    code=prior.sweep.cohort_for('code')[0]
    samples=[dict(sample,code_input=code['input'],code_pct=p) for p in PROPORTIONS]
    streams=[mixed_stream(s,3.) for s in samples]
    assert streams[0][0]==ORIGINAL_GENERATE(sample,3.)
    for j in range(1,5):
        assert [r['arrival_ms'] for r in streams[j][0]]==[r['arrival_ms'] for r in streams[0][0]]
        assert all(not a or b for a,b in zip(streams[j-1][1],streams[j][1]))
        for start in range(0,len(streams[j][1])-3,4):
            assert sum(streams[j][1][start:start+4])==j
    jobs=[(arm,pool.submit(worker,samples[0],2048,arm,2.)) for arm in ('spare','rl_standalone')]
    for arm,f in jobs:
        row=f.result()
        old=prior.read(prior.OUT/'rates'/f'conversation_c2048_{arm}_seed7047_r2.json')
        prior.same_rows([row],[next(r for r in old['rows'] if r['name']==sample['name'])])
    prior.write(OUT/f'smoke_{stage}.json',dict(endpoint_zero_parity=True,common_arrivals=True,
                nested_mixture_assignments=True,balanced_four_arrival_blocks=True))
    print('Smoke: endpoint parity and mixture construction passed',flush=True)


def main(stage,kvs,workers):
    tasks,selected=prepare(stage,kvs)
    started=time.monotonic();done=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        smoke(pool,stage)
        with ThreadPoolExecutor(max_workers=10) as threads:
            futures={threads.submit(scan,t,pool,selected):t for t in tasks}
            for f in as_completed(futures):
                done.append(f.result())
                prior.write(OUT/f'status_{stage}.json',dict(complete=len(done)==len(tasks),
                    groups=len(done),expected=len(tasks),wall_s=time.monotonic()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('mix','kv'),required=True)
    parser.add_argument('--kvs',default='2048,4096')
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args()
    main(args.stage,tuple(map(int,args.kvs.split(','))),args.workers)
