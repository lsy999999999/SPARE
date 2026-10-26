"""One native V1 call, with policy cost separated from physical-plan auditing."""
import time
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus
from research.v1_budget_20260922.scheduler import BudgetScheduler
from research.v1_budget_20260922.policy import State, project
from research.v1_broad12h_20260925.policy import ArrivalRate
from research.www_fixed512_capacity_20261014.policy import select
from research.v1_paper_comparison_20260922.scheduler import ComparisonScheduler
from dataclasses import asdict
import hashlib
import json


class GPUBaselineScheduler(ComparisonScheduler):
    manual_priority_preemption = False
    def configure(self, method, budget, chunk, concurrency, threshold=1, branch=None, evaluation_options=None):
        self.evaluation_options = dict(evaluation_options or {})
        if method != "threshold":
            return ComparisonScheduler.configure(self, method, budget, chunk, concurrency)
        # Native is used only during base-class construction.
        super().configure('native' if method == 'native' else 'srp', budget, chunk, concurrency)
        self.method = method
        self.threshold_spec = threshold
        self.branch_spec = branch
        self.branch_seen = False
        self.deferrals = {}
        self.arrival_clock = getattr(self,'arrival_clock',time.perf_counter)
        self.arrival_origin = self.arrival_clock()
        self.arrival_estimator = ArrivalRate(1.0)
        self.arrival_pending = []
        self.arrival_ids = set()

    def add_request(self,request):
        if self.method != "threshold":
            return super().add_request(request)
        super().add_request(request)
        assert request.request_id not in self.arrival_ids
        self.arrival_ids.add(request.request_id)
        t=self.arrival_clock()-self.arrival_origin
        self.arrival_estimator.observe(t,request.num_prompt_tokens)
        self.arrival_pending.append(dict(id=request.request_id,time_s=t,prompt=request.num_prompt_tokens))

    def schedule(self):
        if self.method != 'threshold':
            return ComparisonScheduler.schedule(self)
        start = time.perf_counter()
        self._order()
        manager = self.kv_cache_manager
        tables = manager.coordinator.single_type_managers[0].req_to_blocks
        free = manager.block_pool.get_num_free_blocks()
        total = self.kv_cache_config.num_blocks
        rows = []
        for running, queue in ((True, self.running), (False, self.waiting)):
            for r in queue:
                assert not r.has_encoder_inputs and not r.use_structured_output and not r.spec_token_ids
                assert r.status in (RequestStatus.RUNNING, RequestStatus.WAITING, RequestStatus.PREEMPTED)
                rows.append(State(r.request_id, r.num_computed_tokens, r.num_tokens,
                    r.num_prompt_tokens, len(r.output_token_ids),
                    len(tables.get(r.request_id, ())), running))
        index = len(self.round_rows)
        recent_rows = self.round_rows[-8:]
        recent = dict(recent_growth=sum(max(0, r['actual_growth_blocks']) for r in recent_rows) / max(len(recent_rows), 1),
                      recent_preemptions=sum(r['preemptions'] for r in recent_rows),
                      recent_release=max(0, free - self.round_rows[-1]['free_after']) if self.round_rows else 0)
        arrival_time_s=self.arrival_clock()-self.arrival_origin
        recent.update(arrival_rate=self.arrival_estimator.at(arrival_time_s),
                      arrival_time_s=arrival_time_s,arrival_observed_tokens=self.arrival_estimator.total)
        arrival_events=self.arrival_pending
        self.arrival_pending=[]
        snapshot = [asdict(r) for r in rows]
        fingerprint = hashlib.sha256(json.dumps(dict(rows=snapshot, free=free, total=total,
            high_water=self.high_water, deferrals=self.deferrals, recent=recent),
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        spec = self.threshold_spec
        branch_applied = False
        if self.branch_spec and index == self.branch_spec['round']:
            assert fingerprint == self.branch_spec['fingerprint'], ('Branch prefix state mismatch', index)
            spec = self.branch_spec['threshold']
            branch_applied = True
            self.branch_seen = True
        choice = select(rows, self.budget, self.chunk, free, total,
                        self.block_size, self.max_num_running_reqs, self.max_model_len,
                        threshold=spec, recent=recent, options=getattr(self, 'evaluation_options', {}))
        budget, chunk = choice['budget'], choice['chunk']
        shadow = False
        self.max_num_scheduled_tokens = budget
        self.scheduler_config.long_prefill_token_threshold = chunk
        decision_ms = (time.perf_counter()-start)*1000
        native_start = time.perf_counter()
        try:
            output = Scheduler.schedule(self)
        finally:
            self.max_num_scheduled_tokens = self.budget
            self.scheduler_config.long_prefill_token_threshold = self.chunk
        native_ms = (time.perf_counter()-native_start)*1000
        audit_start = time.perf_counter()
        args = (free, self.block_size, self.max_num_running_reqs, self.max_model_len)
        base, expected = choice['base'], choice['selected']
        actual = output.num_scheduled_tokens
        after = manager.block_pool.get_num_free_blocks()
        preemptions = sum(r.running and self.requests[r.rid].status==RequestStatus.PREEMPTED for r in rows)
        if expected is not None:
            assert actual == expected.tokens, (actual, expected.tokens)
            assert free-after == expected.growth, (free, after, expected.growth)
            assert preemptions == 0
        if base is not None and self.evaluation_options.get('mode', 'spare') not in ('fixed', 'occupancy'):
            for r in rows:
                if r.rid in base.tokens and ((r.decode and self.evaluation_options.get('protect_decode', True)) or (self.evaluation_options.get('protect_finish', True) and r.outputs==0 and base.tokens[r.rid]==r.remaining)):
                    assert actual.get(r.rid,0) >= base.tokens[r.rid]
        recomputed = 0
        details = []
        for r in rows:
            computed = self.requests[r.rid].num_computed_tokens-actual[r.rid] if r.rid in actual else r.computed
            high = max(self.high_water.get(r.rid,0), r.computed)
            n = actual.get(r.rid,0)
            recomputed += min(n,max(0,high-computed))
            self.high_water[r.rid] = max(high,computed+n)
            if n:
                details.append(dict(id=r.rid,before=computed,tokens=n,prompt=r.prompt,
                    outputs=r.outputs,blocks_before=r.blocks,
                    blocks_after=len(tables.get(r.rid,())),new_admission=not r.running))
        changed = base is not None and actual != base.tokens
        record = dict(round=len(self.round_rows), method=self.method, free_before=free,
            free_after=after,total_blocks=total,usable_blocks=total-1,preemptions=preemptions,
            recomputed_tokens=recomputed,scheduled=dict(actual),allocations=details,
            budget=budget,chunk=chunk,proposed_budget=choice['budget'],
            candidates=choice['candidates'],proposed_change=choice['changed'],
            intervened=bool(changed and not shadow),budget_changed=budget!=self.budget or chunk!=self.chunk,
            recovery_fallback=expected is None,verified=expected is not None,
            decision_ms=decision_ms,native_ms=native_ms,
            baseline=base.tokens if base is not None else None,
            baseline_growth_blocks=base.growth if base is not None else None,
            rejected_guard=choice['rejected_guard'],rejected_saving=choice['rejected_saving'],
            rejected_infeasible=choice['rejected_infeasible'])
        record['reductions'] = [dict(id=r.rid,baseline_tokens=base.tokens[r.rid],
            actual_tokens=actual.get(r.rid,0),computed=r.computed,remaining=r.remaining,
            outputs=r.outputs,running=r.running) for r in rows
            if base is not None and base.tokens.get(r.rid,0)>actual.get(r.rid,0)]
        reduced = {r['id'] for r in record['reductions']}
        self.deferrals = {rid:self.deferrals.get(rid,0)+1 for rid in reduced}
        record.update(pressure_triggered=False, threshold=choice['threshold'],
                      arrival_events=arrival_events,
                      features=choice['features'], state=snapshot, state_fingerprint=fingerprint,
                      branch_applied=branch_applied,
                      candidate_diagnostics=choice.get('candidate_diagnostics', []),
                      actual_growth_blocks=free-after,
                      margin_after_baseline=None if base is None else free-base.growth,
                      max_consecutive_reductions=max(self.deferrals.values(), default=0),
                      consecutive_reductions=dict(self.deferrals))
        record['audit_ms'] = (time.perf_counter()-audit_start)*1000
        record['schedule_ms'] = (time.perf_counter()-start)*1000
        self.round_rows.append(record)
        return output
