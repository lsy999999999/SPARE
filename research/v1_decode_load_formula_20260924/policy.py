"""One added observable: KV held by requests that already emitted a token."""
import math
from research.v1_saving_threshold_20260924.policy import select as old_select


def observable(rows, free, total, block_size):
    remaining=[max(0,r.prompt-r.computed) for r in rows if r.outputs==0]
    work=sum(remaining)
    held=sum(r.blocks for r in rows if r.outputs>0)
    z=held/max(total-1,1)
    assert 0<=z<=1
    return dict(work_free_ratio=free/(1+work/block_size),
                decoded_kv_fraction=z,decoded_kv_headroom=1-z,
                decoded_kv_blocks=held)


def select(rows,budget,chunk,free,total,block_size,concurrency,max_model_len,
           threshold=1,recent=None):
    extra=dict(recent or {})
    extra.update(observable(rows,free,total,block_size))
    return old_select(rows,budget,chunk,free,total,block_size,concurrency,
                      max_model_len,threshold,extra)


def formula(c,a=8,b=128):
    assert b>=0 and c>=0
    return dict(kind='linear',intercept=a,features=['work_free_ratio','decoded_kv_headroom'],
                coefficients=[b,c])
