"""Paper equations on observable states. All times are milliseconds."""
from bisect import bisect_right
from itertools import combinations
import json
from pathlib import Path
import numpy as np


def sampled_peaks(prompts, generated, predictions, block_size=16):
    """Exact physical-block future peaks for each independent sample row."""
    p=np.asarray(prompts,dtype=np.int64)
    g=np.asarray(generated,dtype=np.int64)
    pred=np.asarray(predictions,dtype=np.int64)
    if not len(p):
        return np.zeros(pred.shape[0])
    remaining=pred-g[None,:]
    if np.any(remaining<=0):
        raise ValueError('predictions must exceed generated lengths')
    t=remaining[:,:,None]
    alive=remaining[:,None,:]>=t
    blocks=(p[None,None,:]+g[None,None,:]+t+block_size-1)//block_size
    return np.max(np.sum(blocks*alive,axis=2),axis=1)


def conditional_draws(history, generated, caps, draws, rng):
    history=np.sort(np.asarray(history,dtype=np.int64))
    out=np.empty((draws,len(generated)),dtype=np.int64)
    for j,(g,cap) in enumerate(zip(generated,caps)):
        first=np.searchsorted(history,g,side='right')
        if first<len(history):
            out[:,j]=np.minimum(history[rng.integers(first,len(history),size=draws)],cap)
        else:
            out[:,j]=cap
        out[:,j]=np.maximum(g+1,out[:,j])
    return out


def features(allocations):
    pre=[a for a in allocations if not (a['outputs']>0 and a['tokens']==1 and a['before']+1==a['prompt']+a['outputs'])]
    dec=[a for a in allocations if a not in pre]
    p=[sum(a['before']*a['tokens'] for a in pre),sum(a['tokens']**2 for a in pre),sum(a['tokens'] for a in pre),int(bool(pre))]
    d=[len(dec),sum(a['before'] for a in dec),int(bool(dec))]
    return p,d


class CostModel:
    def __init__(self, model):
        self.p=np.asarray(model['prefill_coefficients'],dtype=float)
        self.d=np.asarray(model['decode_coefficients'],dtype=float)
        self.gamma=1.

    def predict(self, allocations, scaled=True):
        p,d=features(allocations)
        factor=self.gamma if scaled else 1.
        return factor*float(np.dot(self.p,p)),factor*float(np.dot(self.d,d))

    def update(self, allocations, observed_ms):
        raw=sum(self.predict(allocations,scaled=False))
        if raw>0 and observed_ms>0:
            self.gamma=.1*(observed_ms/raw)+.9*self.gamma


def nonnegative_fit(x,y):
    """Small exact active-set NNLS, enumerating at most 16 supports."""
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    scale=np.maximum(np.linalg.norm(x,axis=0),1.)
    z=x/scale
    best=(float(np.dot(y,y)),np.zeros(x.shape[1]))
    for count in range(1,x.shape[1]+1):
        for subset in combinations(range(x.shape[1]),count):
            coeff=np.linalg.lstsq(z[:,subset],y,rcond=None)[0]
            if np.any(coeff<0):
                continue
            full=np.zeros(x.shape[1]);full[list(subset)]=coeff
            error=float(np.sum((z@full-y)**2))
            if error<best[0]:best=(error,full)
    return (best[1]/scale).tolist()


def fit_profile(profile):
    train={'p':[],'d':[]}; validation=[]
    for episode in profile:
        for row in episode['rounds']:
            p,d=features(row['allocations'])
            # First round includes request/engine warm-up bookkeeping; calibration
            # repeats begin in a warm engine and retain it, as does formal timing.
            if episode['repeat']==2:
                validation.append((p,d,row['step_ms']))
            elif p[-1] and not d[-1]:train['p'].append((p,row['step_ms']))
            elif d[-1] and not p[-1]:train['d'].append((d,row['step_ms']))
    model={}
    for key,label in [('p','prefill'),('d','decode')]:
        x,y=zip(*train[key]);model[label+'_coefficients']=nonnegative_fit(x,y)
        model[label+'_training_rows']=len(x)
    errors=[]; abs_errors=[]
    for p,d,y in validation:
        pred=float(np.dot(p,model['prefill_coefficients'])+np.dot(d,model['decode_coefficients']))
        errors.append(abs(pred-y)/y);abs_errors.append(abs(pred-y))
    model.update(validation_rows=len(validation),validation_mape=float(np.mean(errors)),
        validation_mae_ms=float(np.mean(abs_errors)),alpha=.1,
        calibration='fresh RTX 4090 synthetic inputs, repeat 0/1 train, repeat 2 validation',
        equations='SOLA Eq.3/4; nonnegative coefficients; Eq.5 online EMA',
        feature_units='tokens, milliseconds',target_slos_ms=dict(ttft=5000,tpot=200))
    return model
