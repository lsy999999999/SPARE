"""One native V1 call, with policy cost separated from physical-plan auditing."""
import time
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus
from research.v1_budget_20260922.scheduler import BudgetScheduler
from research.v1_budget_20260922.policy import State, project
from research.v1_capacity_ablation_20260923.policy import METHODS
from research.v1_state_aware_20260923.policy import NEW, select


class StateScheduler(BudgetScheduler):
    def configure(self, method, budget, chunk, concurrency, occupancy_threshold=.9, pressure="1/4", saving="1/2"):
        assert method in METHODS + NEW
        super().configure('native' if method=='native' else 'srp', budget, chunk, concurrency)
        self.method = method
        self.occupancy_threshold = occupancy_threshold
        self.pressure, self.saving = pressure, saving
        self.deferrals = {}

    def schedule(self):
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
        choice = select(self.method, rows, self.budget, self.chunk, free, total,
                        self.block_size, self.max_num_running_reqs, self.max_model_len,
                        self.occupancy_threshold, self.pressure, self.saving)
        shadow = self.method == 'a_shadow'
        budget = self.budget if shadow else choice['budget']
        chunk = self.chunk if shadow else choice['chunk']
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
        if choice['projected']:
            base = choice['base']
            expected = base if shadow else choice['selected']
        else:
            base = project(rows, self.budget, self.chunk, *args)
            expected = base if (budget,chunk)==(self.budget,self.chunk) else project(rows,budget,chunk,*args)
        actual = output.num_scheduled_tokens
        after = manager.block_pool.get_num_free_blocks()
        preemptions = sum(r.running and self.requests[r.rid].status==RequestStatus.PREEMPTED for r in rows)
        if expected is not None:
            assert actual == expected.tokens, (actual, expected.tokens)
            assert free-after == expected.growth, (free, after, expected.growth)
            assert preemptions == 0
        if base is not None and self.method in ('a_frozen','a_shadow','a_any_positive_saving',
                                                'a_no_finish_guard','occupancy_finish_guard') + NEW:
            for r in rows:
                if r.rid in base.tokens and (r.decode or (self.method!='a_no_finish_guard'
                        and r.outputs==0 and base.tokens[r.rid]==r.remaining)):
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
        record.update(pressure_triggered=choice.get('pressure_triggered', False),
                      candidate_diagnostics=choice.get('candidate_diagnostics', []),
                      actual_growth_blocks=free-after,
                      margin_after_baseline=None if base is None else free-base.growth,
                      max_consecutive_reductions=max(self.deferrals.values(), default=0),
                      consecutive_reductions=dict(self.deferrals))
        record['audit_ms'] = (time.perf_counter()-audit_start)*1000
        record['schedule_ms'] = (time.perf_counter()-start)*1000
        self.round_rows.append(record)
        return output
