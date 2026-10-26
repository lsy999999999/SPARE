"""Numerical checks of derived identities, thresholds and counterexamples (not latency simulation)."""
from pathlib import Path
import math
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_theory_formula_20260924.model import benefit,integer_threshold,discrete_benefit
from research.v1_theory_formula_20260924.policy import values,arms
from research.v1_theory_formula_20260924.common import OUT,write
from research.v1_saving_threshold_20260924.policy import threshold_value


def main():
    nodes,weights=np.polynomial.legendre.leggauss(256)
    checks=0;integrals=0;max_error=0.
    for family in ('lomax','exponential'):
        for q in (1,16,64,128,256,1024,4096,16384):
            m=q/16
            for free in (0,1,16,64,512):
                for d in (1,8,32,64):
                    t=free+d*(nodes+1)/2
                    surv=m/(m+t) if family=='lomax' else np.exp(-t/m)
                    numeric=float(np.dot(weights,surv)*d/2)
                    exact=benefit(family,free,d,m)
                    err=abs(numeric-exact)/max(abs(exact),1e-12)
                    assert err<1e-7,(family,q,free,d,err)
                    max_error=max(max_error,err);integrals+=1
                for cost in (1,8,24,40):
                    closed=integer_threshold(family,free,m,cost,64)
                    # Log domain preserves the strict finite-d gap below an exponential
                    # asymptote; ordinary subtraction rounds 1-exp(-38) to 1.
                    def sufficient(d):
                        if family=='exponential':
                            log_b=math.log(m)-free/m+math.log1p(-math.exp(-d/m)) if d/m>1 else math.log(m)-free/m+math.log(-math.expm1(-d/m))
                            return log_b>=math.log(cost)
                        return benefit(family,free,d,m)>=cost
                    brute=next((d for d in range(1,65) if sufficient(d)),math.inf)
                    assert closed==brute,(family,q,free,cost,closed,brute)
                    checks+=1
    # Integer-valued KV demands: exact tail SUM, distinct from the continuous examples.
    discrete_checks=0
    ys=[0,2,8,32,128];ps=[.1,.2,.25,.3,.15]
    for free in range(20):
        for d in range(1,65):
            direct=discrete_benefit(free,d,ys,ps)
            tails=sum(sum(p for y,p in zip(ys,ps) if y>=free+j) for j in range(1,d+1))
            assert math.isclose(direct,tails,abs_tol=1e-10)
            discrete_checks+=1
    # Formula expansion and a necessary threshold versus a sufficient exact threshold.
    formula_checks=0
    for q in (1,16,64,128,256,1024,4096,16384):
        for free in (0,1,16,64,512):
            for z in (0,.25,.5,.75,1):
                f=values(free,(q-1)*16,z*1023,1024)
                ns={name:threshold_value(spec,f) for name,spec in arms().items()}
                assert ns['lomax']>=ns['old'] and ns['exponential']>=ns['old']
                assert ns['coupled_exact']>=ns['coupled_decode']>=ns['decode']
                x=free/q;u=1-z
                assert math.isclose((8+32*u)*(1+16*x)-(8+128*x+32*u),512*x*u,abs_tol=1e-9)
                formula_checks+=1
    # TTFT is exactly the area under the number of arrived, not-yet-first-token requests.
    rng=np.random.default_rng(20260924)
    area_checks=0;policy_checks=0;regret_checks=0
    for _ in range(100):
        arrival=rng.uniform(0,10,20);first=arrival+rng.uniform(.01,12,20)
        times=sorted(set([*arrival,*first]))
        area=sum((b-a)*int(np.sum((arrival<=(a+b)/2)&(first>(a+b)/2))) for a,b in zip(times,times[1:]))
        assert math.isclose(area,float(np.sum(first-arrival)),rel_tol=1e-12)
        area_checks+=1
        # Finite absorbing MDP: direct policy evaluation equals summed baseline advantages.
        costs=rng.uniform(0,3,(5,3,2))
        trans=rng.dirichlet(np.ones(3),size=(5,3,2))
        v0=np.zeros((6,3))
        for t in reversed(range(5)):v0[t]=costs[t,:,0]+trans[t,:,0]@v0[t+1]
        qa=np.stack([costs[t]+np.einsum('sak,k->sa',trans[t],v0[t+1]) for t in range(5)])
        policy=np.argmin(qa,axis=2)
        vp=np.zeros((6,3))
        for t in reversed(range(5)):
            for s in range(3):
                a=policy[t,s];vp[t,s]=costs[t,s,a]+trans[t,s,a]@vp[t+1]
        occ=np.array([1.,0.,0.]);adv=0.
        for t in range(5):
            nxt=np.zeros(3)
            for s in range(3):
                a=policy[t,s];adv+=occ[s]*(qa[t,s,a]-v0[t,s]);nxt+=occ[s]*trans[t,s,a]
            occ=nxt
        assert math.isclose(adv,vp[0,0]-v0[0,0],abs_tol=1e-12)
        assert vp[0,0]<=v0[0,0]+1e-12
        policy_checks+=1
        truth=rng.uniform(0,20,5);eps=.7;eta=.3
        estimate=truth+rng.uniform(-eps,eps,5)
        eligible=np.flatnonzero(estimate<=min(estimate)+eta)
        chosen=int(rng.choice(eligible))
        assert truth[chosen]-min(truth)<=2*eps+eta+1e-12
        regret_checks+=1
    counterexamples=dict(
        same_observation_opposite_future=dict(saving=16,cost=8,no_shortage_net_gain=-8,shortage32_net_gain=8),
        candidate_dependent_cost=dict(savings=[16,32],costs=[8,40],net_gains=[8,-8]),
        decode_release_sign=dict(free=100,cost=8,saving=16,decode_blocks=[100,300],
                                 demand_after_release=[110,90],net_gains=[2,-8]))
    write(OUT/'model_checks.json',dict(passed=True,continuous_threshold_cases=checks,
        quadrature_cases=integrals,max_quadrature_relative_error=max_error,discrete_tail_cases=discrete_checks,
        formula_cases=formula_checks,ttft_area_cases=area_checks,policy_improvement_cases=policy_checks,
        regret_cases=regret_checks,counterexamples=counterexamples,
        scope='Checks algebra and model logic only; no GPU latency, estimated real-world risk or runtime guarantee.'))
    print('PASS:',checks,'threshold cases,',integrals,'integrals,',discrete_checks,'integer-tail identities,',formula_checks,'formula cases; 100 cases each for TTFT area, policy improvement and regret.')


if __name__=='__main__':main()
