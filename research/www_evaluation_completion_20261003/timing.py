"""Retrospective GPU-trace calibration, frozen by source-window identity."""
import csv
import gzip
import math
import statistics
import numpy as np
from scipy.optimize import nnls
from .common import GPU, OUT, ROOT, read, write, digest

NAMES = ['base', 'prefill_k', 'prefill_attention_m', 'decode_32', 'decode_context_100k',
         'token_256_tiles', 'has_prefill', 'mixed']


def vector(pre, contexts, dec):
    return [1., sum(pre)/1024, sum(n*c+n*(n+1)/2 for n,c in zip(pre,contexts))/1e6,
            len(dec)/32, sum(dec)/1e5, math.ceil((sum(pre)+len(dec))/256),
            float(bool(pre)), float(bool(pre) and bool(dec))]


def batch_row(row):
    pre, contexts, dec = [], [], []
    state = {r['rid']: r for r in row.get('state', [])}
    for a in row['allocations']:
        # Reconstruct tokens present before the iteration without target output length.
        remaining = a['prompt']+a['outputs']-a['before']
        if a['outputs'] > 0 and a['tokens'] == 1 and remaining == 1:
            dec.append(a['before'])
        else:
            pre.append(a['tokens']); contexts.append(a['before'])
    return vector(pre, contexts, dec)


class FittedTiming:
    def __init__(self, payload):
        self.coefficients = payload['coefficients']
    def estimate(self, batch):
        values = vector(batch.prefill_chunks, batch.prefill_contexts, batch.decode_contexts)
        return max(.1, sum(a*b for a,b in zip(values,self.coefficients))), ('retrospective_gpu_fit',)


def gpu_runs():
    return list(csv.DictReader((GPU/'analysis/runs.csv').open()))


def round_path(run):
    result = read(GPU/run['result_path'])
    return (GPU/run['result_path']).parent.parent/result['rounds_path'], result


def calibrate():
    target = OUT/'timing_profile.json'
    if target.exists():
        return read(target)
    rows = gpu_runs()
    train = [r for r in rows if 'fresh0' in r['cell'] and r['arm'] in ('arrival','srp','native') and int(r['repeat']) < 2]
    x, y, sources, train_ids = [], [], [], set()
    for run in train:
        path, result = round_path(run)
        assert digest(GPU/run['result_path']) == run['sha256']
        assert digest(path) == result['rounds_sha256']
        train_ids.update(r['request_id'] for r in result['requests'])
        with gzip.open(path,'rt') as h:
            records = __import__('json').load(h)
        for row in records:
            if row['allocations']:
                x.append(batch_row(row)); y.append(row['step_ms'])
        sources.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path)))
    coefs, norm = nnls(np.asarray(x), np.asarray(y))
    profile = dict(kind='nonnegative_batch_affine',features=NAMES,coefficients=coefs.tolist(),
                   training_episodes=len(train),training_iterations=len(y),training_request_ids=sorted(train_ids),
                   sources=sources,training_rmse_ms=norm/math.sqrt(len(y)),
                   scope='Retrospective fit: GPU r3 fresh0, arrival/SRP/native, repeats 0-1; total engine.step wall time including instrumentation.',
                   holdout='GPU r3 fresh1, never used to fit or choose this model; historical research windows, not prospectively collected validation.',
                   performance_validated=False)
    write(target, profile)  # Freeze before loading holdout timing labels.
    errors = []
    holdout_ids = set()
    for run in rows:
        if 'fresh1' not in run['cell']:
            continue
        path,result=round_path(run)
        assert digest(GPU/run['result_path']) == run['sha256']
        assert digest(path) == result['rounds_sha256']
        holdout_ids.update(r['request_id'] for r in result['requests'])
        with gzip.open(path,'rt') as h:
            records=__import__('json').load(h)
        delta=[];relative=[]
        for row in records:
            if not row['allocations']:continue
            pred=max(.1,sum(a*b for a,b in zip(batch_row(row),coefs)))
            delta.append(pred-row['step_ms']);relative.append(abs(pred-row['step_ms'])/row['step_ms'])
        errors.append(dict(cell=run['cell'],arm=run['arm'],repeat=int(run['repeat']),
                           n=len(delta),mae_ms=statistics.fmean(map(abs,delta)),
                           bias_ms=statistics.fmean(delta),mape_pct=100*statistics.fmean(relative)))
    assert not train_ids & holdout_ids
    write(OUT/'timing_step_validation.json',dict(complete=True,request_id_overlap=0,
          independent_new_collection=False,profile_sha256=digest(target),episodes=errors))
    return profile
