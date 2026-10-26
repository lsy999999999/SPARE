"""A 2x2 factorial experiment: causal prompt-arrival rate and decoded KV occupancy."""
import math
from research.v1_saving_threshold_20260924.policy import select as old_select


class ArrivalRate:
    def __init__(self,horizon_s=1.0):
        assert horizon_s>0
        self.horizon_s=horizon_s
        self.value=0.0
        self.last=0.0
        self.total=0

    def at(self,t):
        assert t>=self.last,(t,self.last)
        self.value*=math.exp(-(t-self.last)/self.horizon_s)
        self.last=t
        return self.value

    def observe(self,t,prompt):
        assert prompt>0
        self.at(t)
        self.value+=prompt/self.horizon_s
        self.total+=prompt


def formula(tau_s,decode_coefficient=0,mode="decode"):
    assert mode in ("decode","pressure_decode","pressure_constant")
    assert tau_s>=0 and decode_coefficient>=0
    return dict(kind='linear',intercept=8,features=['arrival_adjusted_ratio',{'decode':'decoded_kv_headroom','pressure_decode':'pressure_decoded_headroom','pressure_constant':'pressure_gate'}[mode]],
                coefficients=[128,decode_coefficient],tau_s=tau_s)


def select(rows,budget,chunk,free,total,block_size,concurrency,max_model_len,
           threshold=1,recent=None):
    extra=dict(recent or {})
    work=sum(max(0,r.prompt-r.computed) for r in rows if r.outputs==0)
    rate=extra.get('arrival_rate',0.)
    held=sum(r.blocks for r in rows if r.outputs>0)
    z=held/max(total-1,1)
    assert 0<=z<=1
    tau=threshold.get('tau_s',0.) if isinstance(threshold,dict) else 0.
    assert rate>=0 and tau>=0
    x=free/(1+(work+tau*rate)/block_size)
    gate=x/(1+x)
    extra.update(pressure_gate=gate,pressure_decoded_headroom=(1-z)*gate,work_free_ratio=free/(1+work/block_size),
                 arrival_adjusted_ratio=free/(1+(work+tau*rate)/block_size),
                 arrival_expected_tokens=tau*rate,arrival_tau_s=tau,
                 decoded_kv_fraction=z,decoded_kv_headroom=1-z,decoded_kv_blocks=held)
    return old_select(rows,budget,chunk,free,total,block_size,concurrency,
                      max_model_len,threshold,extra)
