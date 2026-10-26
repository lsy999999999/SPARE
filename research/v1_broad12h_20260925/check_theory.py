"""Numerical checks of stated conditional identities, not GPU performance claims."""
import math,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_broad12h_20260925.common import *

def main():
    rng=random.Random(20260925);counts=dict(regime_threshold=0,ttft_area=0,finite_gap_bound=0,gate_properties=0)
    for _ in range(500):
        x=10**rng.uniform(-4,2);a=rng.uniform(1,20);b=rng.uniform(1,200);k=rng.uniform(0,50);v=rng.uniform(.1,10)
        p=x/(1+x);n=a+b*x+k*p
        def gain(d):return (1-p)*v*(d-a-b*x)+p*v*(d-a-b*x-k)
        assert math.isclose(gain(n),0,abs_tol=1e-8)
        ni=math.ceil(n);assert gain(ni)>=-1e-8 and gain(ni-1)<1e-8
        counts['regime_threshold']+=1
        assert 0<=p<=1 and p<=x and a+b*x<=n<=a+b*x+k
        eps=1e-5*max(1,x);fd=((a+b*(x+eps)+k*(x+eps)/(1+x+eps))-(a+b*x+k*x/(1+x)))/eps
        assert math.isclose(fd,b+k/(1+x)**2,rel_tol=2e-4,abs_tol=1e-3)
        counts['gate_properties']+=1
        Q=10**rng.uniform(1,4);F=rng.uniform(0,Q);d=rng.uniform(.1,50);beta=rng.uniform(.1,20);xx=F/Q
        exact=Q/beta*math.log1p(beta*d/(Q+beta*F));linear=d/(1+beta*xx)
        assert -1e-8<=linear-exact<=beta*d*d/(2*Q)+1e-8
        counts['finite_gap_bound']+=1
        reqs=[(rng.uniform(0,20),rng.uniform(.1,10)) for _ in range(20)]
        events=sorted([(t,1) for t,w in reqs]+[(t+w,-1) for t,w in reqs]);prev=events[0][0];u=0;area=0
        for t,delta in events:area+=u*(t-prev);u+=delta;prev=t
        assert u==0 and math.isclose(area,sum(w for t,w in reqs),abs_tol=1e-8)
        counts['ttft_area']+=1
    # Empirical EWMA at fixed events obeys exact no-arrival exponential decay.
    for H in (.1,1,5):
        for dt in (0,.1,1,10):
            events=[(0,100),(1,700),(2,300)];t=3
            rate=sum(p/H*math.exp(-(t-ti)/H) for ti,p in events)
            after=sum(p/H*math.exp(-(t+dt-ti)/H) for ti,p in events)
            assert math.isclose(after,rate*math.exp(-dt/H),rel_tol=1e-10,abs_tol=1e-10)
    write(OUT/'theory_checks.json',dict(passed=True,counts=counts,note='Analytic model identities checked numerically; assumptions not established for real GPU.'))
    print('Theory checks passed:',counts)

if __name__=='__main__':main()
