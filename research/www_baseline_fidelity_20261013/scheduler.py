"""Expanded paper mechanisms with one native V1 call and physical-plan audit."""
from collections import deque
import json
import math
import time
import numpy as np
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.request import RequestStatus
from research.www_evaluation_gpu_completion_20261003.scheduler import GPUBaselineScheduler as Base
from research.v1_budget_20260922.scheduler import BudgetScheduler
from research.v1_budget_20260922.policy import State, project
from research.v1_paper_comparison_20260922.rules import future_peak_blocks
from research.www_baseline_fidelity_20261013.policy import CostModel, conditional_draws, sampled_peaks


class GPUBaselineScheduler(Base):
    NEW=('past_future_multi','sola_calibrated')

    def configure(self, method, budget, chunk, concurrency, **kwargs):
        self.fidelity_method=method if method in self.NEW else None
        super().configure('native' if self.fidelity_method else method,budget,chunk,concurrency,**kwargs)
        self.rng=np.random.default_rng(getattr(self,'seed_context',20260929))
        self.seen_at={};self.first_at={};self.latest_at={}
        self.phase_history=deque(maxlen=8)
        self.length_groups={}
        self.cost=None
        if method=='sola_calibrated':
            path=getattr(self,'fidelity_cost_path',None)
            model=getattr(self,'fidelity_cost_model',None)
            if model is None:
                if path is None:raise ValueError('SOLA requires independently calibrated cost model')
                model=json.loads(path.read_text())
            self.cost=CostModel(model)
        self.observed_round=None

    def _now(self):
        return getattr(self,'arrival_clock',time.perf_counter)()*1000

    def add_request(self,request):
        super().add_request(request)
        if self.fidelity_method:
            # Engine arrival_time uses wall time. Keep one local monotonic clock
            # for all observed ages; never subtract it from a wall-clock value.
            self.seen_at[request.request_id]=self._now()

    def _free_request(self,request):
        if self.fidelity_method and request.status in (RequestStatus.FINISHED_STOPPED,RequestStatus.FINISHED_LENGTH_CAPPED):
            self.predictor.observe(len(request.output_token_ids))
            group=min(4,request.num_prompt_tokens//1024)
            self.length_groups.setdefault(group,deque(maxlen=1000)).append(len(request.output_token_ids))
        return super()._free_request(request)

    def observe_step(self,unused=None):
        if not self.fidelity_method or not self.round_rows:return
        row=self.round_rows[-1]
        if row is self.observed_round:return
        now=self._now()
        for a in row['allocations']:
            if a['before']+a['tokens']==a['prompt']+a['outputs']:
                self.first_at.setdefault(a['id'],now)
                self.latest_at[a['id']]=now
        if self.cost and 'step_ms' in row:
            self.cost.update(row['allocations'],row['step_ms'])
        self.observed_round=row

    def _states(self,requests):
        tables=self.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
        running={r.request_id for r in self.running}
        return [State(r.request_id,r.num_computed_tokens,r.num_tokens,r.num_prompt_tokens,
                      len(r.output_token_ids),len(tables.get(r.request_id,())),r.request_id in running)
                for r in requests]

    def _order(self):
        if not self.fidelity_method:return super()._order()
        if self.fidelity_method=='past_future_multi':return self._past_future()
        return self._sola()

    def _past_future(self):
        # Same FIFO gate, physical capacity, history and native recovery as old PF.
        decode=[r for r in self.running if len(r.output_token_ids)>0 and r.num_tokens-r.num_computed_tokens==1]
        ids={r.request_id for r in decode}
        self.running[:]=decode+[r for r in self.running if r.request_id not in ids]
        free=self.kv_cache_manager.block_pool.get_num_free_blocks()
        rows=self._states(self.running)
        plan=project(rows,self.budget,self.chunk,free,self.block_size,self.max_num_running_reqs,self.max_model_len)
        self.fidelity_audit=dict(pf_samples=0,pf_gate_checked=0,pf_gate_rejected=0)
        if plan is None or not self.waiting:return
        remaining_budget=self.budget-sum(plan.tokens.values())
        slots=self.max_num_running_reqs-len(self.running)
        if remaining_budget<=0 or slots<=0:return
        repeats=32 if len(self.running)<16 else 1
        self.fidelity_audit['pf_samples']=repeats
        active=list(self.running)
        prompts=[r.num_prompt_tokens for r in active]
        generated=[len(r.output_token_ids) for r in active]
        draws=conditional_draws(self.predictor.history,generated,[self.max_model_len-p for p in prompts],repeats,self.rng)
        remaining_free=free-plan.growth
        allowed=[];waiting=list(self.waiting)
        for request in waiting:
            if slots<=0 or remaining_budget<=0:break
            row=self._states([request])[0]
            count=min(row.remaining,remaining_budget,self.chunk)
            growth=max(0,math.ceil((row.computed+count)/16)-row.blocks)
            if growth>remaining_free:break
            new=conditional_draws(self.predictor.history,[row.outputs],[self.max_model_len-row.prompt],repeats,self.rng)
            proposed=np.concatenate((draws,new),axis=1)
            peaks=sampled_peaks(prompts+[row.prompt],generated+[row.outputs],proposed)
            mean=float(np.mean(peaks))
            self.fidelity_audit.update(pf_peak_mean=mean,pf_peak_std=float(np.std(peaks)))
            self.fidelity_audit['pf_gate_checked']+=1
            if mean>self.kv_cache_config.num_blocks-1:
                self.fidelity_audit['pf_gate_rejected']+=1
                break
            allowed.append(request);draws=proposed
            prompts.append(row.prompt);generated.append(row.outputs)
            remaining_budget-=count;remaining_free-=growth;slots-=1
        self.fidelity_waiting=waiting[len(allowed):]
        self.waiting=FCFSRequestQueue(allowed)

    def _sola(self):
        now=self._now()
        all_requests=list(self.running)+list(self.waiting)
        rows={s.rid:s for s in self._states(all_requests)}
        requests={r.request_id:r for r in all_requests}
        pre=[s for s in rows.values() if not s.decode]
        dec=[s for s in rows.values() if s.decode]
        age=lambda s:max(0.,now-self.seen_at[s.rid])
        tpot=lambda s:max(0.,now-self.first_at.get(s.rid,now))/max(1,s.outputs-1)
        allocation=lambda s,n:dict(id=s.rid,before=s.computed,tokens=n,prompt=s.prompt,outputs=s.outputs)
        prefill_costs={s.rid:self.cost.predict([allocation(s,max(1,s.remaining))])[0] for s in pre}
        full_cost=lambda s:prefill_costs[s.rid]
        lengths={}
        for s in rows.values():
            pool=self.length_groups.get(min(4,s.prompt//1024),self.predictor.history)
            conditional=[x for x in pool if x>s.outputs]
            lengths[s.rid]=max(s.outputs+1,min(self.max_model_len-s.prompt,
                int(np.median(conditional)) if conditional else self.max_model_len-s.prompt))
        decode_cost=self.cost.predict([allocation(s,1) for s in dec])[1]
        pre.sort(key=lambda s:(-(age(s)+full_cost(s)),s.rid))
        dec.sort(key=lambda s:(-((tpot(s)*s.outputs+decode_cost*(lengths[s.rid]-s.outputs))/lengths[s.rid]),s.rid))
        ppressure=max((age(s)/5000 for s in pre),default=0.)
        dpressure=max((tpot(s)/200 for s in dec),default=0.)
        self.phase_history.append(ppressure-dpressure)
        phase='prefill' if sum(self.phase_history)>=0 else 'decode'
        free=self.kv_cache_manager.block_pool.get_num_free_blocks()
        resident=[s for s in rows.values() if s.running]
        capacity=self.kv_cache_config.num_blocks-1
        original_concurrency=self.max_num_running_reqs
        forecast_cache={}

        def candidate(budget,chunk,npref):
            selected=[];remaining=budget
            order=(pre[:npref]+dec) if phase=='prefill' else (dec+pre[:npref])
            admitted=[]
            for s in order:
                if remaining<=0:break
                if not s.running:
                    if len(resident)+len(admitted)>=original_concurrency:continue
                    forecast=resident+admitted+[s]
                    key=tuple(r.rid for r in admitted+[s])
                    if key not in forecast_cache:
                        forecast_cache[key]=future_peak_blocks([(r.prompt,r.outputs,lengths[r.rid]) for r in forecast])
                    if forecast_cache[key]>capacity:continue
                n=min(s.remaining,chunk,remaining)
                if n<=0:continue
                selected.append(s);remaining-=n
                if not s.running:admitted.append(s)
            # Native running/waiting ordering is retained. Held running requests
            # continue to own their KV blocks and consume concurrency slots.
            ordered=[s for s in selected if s.running]+[s for s in selected if not s.running]
            held=sum(s.running for s in rows.values())-sum(s.running for s in selected)
            plan=project(ordered,budget,chunk,free,16,original_concurrency-held,self.max_model_len)
            if plan is None or not plan.tokens:return None
            alloc=[allocation(rows[rid],n) for rid,n in plan.tokens.items()]
            cp,cd=self.cost.predict(alloc)
            chosen=set(plan.tokens)
            if phase=='prefill':
                constraints=[tpot(s)+cp/max(1,s.outputs) for s in dec]
                limit=200.
            else:
                constraints=[age(s)+cp+cd+full_cost(s) for s in pre
                             if s.rid not in chosen or plan.tokens[s.rid]<s.remaining]
                limit=5000.
            # Paper permits relaxing to attained percentile when both SLOs fail.
            quantile=1.
            if ppressure>1 and dpressure>1 and constraints:
                current=[tpot(s) for s in dec] if phase=='prefill' else [age(s) for s in pre]
                quantile=sum(x<=limit for x in current)/max(1,len(current))
            constraint=float(np.quantile(constraints,quantile)) if constraints else 0.
            return dict(budget=budget,chunk=chunk,selected=selected,plan=plan,
                        cp=cp,cd=cd,feasible=constraint<=limit,constraint=constraint,
                        relaxed_quantile=quantile,held=held,
                        prefill_tokens=sum(a['tokens'] for a in alloc if not rows[a['id']].decode))

        candidates=[]
        if phase=='prefill':
            for budget in range(64,self.budget+1,64):
                c=candidate(budget,min(self.chunk,budget),len(pre))
                if c:candidates.append(c)
            valid=[c for c in candidates if c['feasible']]
            chosen=max(valid,key=lambda c:(c['prefill_tokens'],sum(c['plan'].tokens.values())),default=None)
        else:
            chosen=None
            for n in range(len(pre)+1):
                c=candidate(self.budget,self.chunk,n)
                if c:
                    candidates.append(c)
                    if c['feasible']:
                        chosen=c;break
        fallback=None
        if chosen is None and candidates:
            chosen=min(candidates,key=lambda c:(c['constraint'],c['cp']+c['cd']))
            fallback='unsatisfied_constraint_minimum_violation'
        if chosen is None:
            # Forecast underestimation/overestimation must not deadlock the
            # shared backend. Its normal preemption path records all KV changes.
            self.fidelity_audit=dict(sola_phase=phase,sola_fallback='native_progress',sola_gamma=self.cost.gamma)
            return
        ids={s.rid for s in chosen['selected']}
        self.fidelity_running=[r for r in self.running if r.request_id not in ids]
        self.fidelity_waiting=[r for r in self.waiting if r.request_id not in ids]
        self.running[:]=[requests[s.rid] for s in chosen['selected'] if s.running]
        self.waiting=FCFSRequestQueue([requests[s.rid] for s in chosen['selected'] if not s.running])
        self.budget=chosen['budget'];self.chunk=chosen['chunk']
        self.max_num_scheduled_tokens=self.budget
        self.scheduler_config.long_prefill_token_threshold=self.chunk
        self.max_num_running_reqs=original_concurrency-chosen['held']
        self.fidelity_audit=dict(sola_phase=phase,sola_gamma=self.cost.gamma,
            sola_predicted_prefill_ms=chosen['cp'],sola_predicted_decode_ms=chosen['cd'],
            sola_constraint=chosen['constraint'],sola_constraint_satisfied=chosen['feasible'],
            sola_relaxed_quantile=chosen['relaxed_quantile'],sola_candidates=len(candidates),
            sola_fallback=fallback,sola_ttft_pressure=ppressure,sola_tpot_pressure=dpressure,
            sola_held_running=len(self.fidelity_running),sola_held_waiting=len(self.fidelity_waiting))

    def schedule(self):
        if not self.fidelity_method:return super().schedule()
        self.observe_step()
        start=time.perf_counter()
        settings=self.budget,self.chunk,self.max_num_running_reqs
        self.fidelity_running=[];self.fidelity_waiting=[];self.fidelity_audit={}
        try:
            result=BudgetScheduler.schedule(self)
        finally:
            self.running.extend(self.fidelity_running)
            self.waiting.extend(self.fidelity_waiting)
            self.budget,self.chunk,self.max_num_running_reqs=settings
            self.max_num_scheduled_tokens=self.budget
            self.scheduler_config.long_prefill_token_threshold=self.chunk
        self.round_rows[-1].update(self.fidelity_audit,method=self.fidelity_method,
                                  schedule_ms=(time.perf_counter()-start)*1000)
        return result
