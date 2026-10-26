"""Single native V1 scheduling call with cheap, audited budget selection."""
import time

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import RequestStatus

from research.v1_budget_20260922.policy import State, choose, project


class BudgetScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        groups = self.kv_cache_config.kv_cache_groups
        if (len(groups) != 1 or type(groups[0].kv_cache_spec) is not FullAttentionSpec
                or groups[0].kv_cache_spec.sliding_window is not None
                or self.cache_config.enable_prefix_caching or self.lora_config
                or self.connector or self.vllm_config.speculative_config
                or self.parallel_config.pipeline_parallel_size != 1
                or self.scheduler_config.policy != 'fcfs'):
            raise ValueError('Requires one full-attention group, no APC/spec/LoRA/P-D/PP')
        self.configure('native', 2048, 2048, 64)

    def configure(self, method, budget, chunk, concurrency):
        assert method in ('native', 'srp', 'budget', 'shadow')
        assert not self.requests
        assert 0 < concurrency <= 64 and 0 < budget <= 2048 and 0 < chunk <= budget
        self.method = method
        self.budget, self.chunk = budget, chunk
        self.max_num_scheduled_tokens = budget
        self.max_num_running_reqs = concurrency
        self.scheduler_config.long_prefill_token_threshold = chunk
        self.round_rows = []
        self.high_water = {}

    def _order(self):
        if self.method == 'native':
            return
        def key(r):
            return (r.num_tokens - r.num_computed_tokens, r.arrival_time, r.request_id)
        # Running/waiting boundary is native. Decode before partial prefill;
        # recomputing previously decoded requests belong to the partial group.
        decode = [r for r in self.running
                  if len(r.output_token_ids) > 0 and r.num_tokens - r.num_computed_tokens == 1]
        decode_ids = {r.request_id for r in decode}
        self.running[:] = decode + sorted(
            (r for r in self.running if r.request_id not in decode_ids), key=key)
        self.waiting = FCFSRequestQueue(sorted(self.waiting, key=key))

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
        choice = None
        if self.method in ('budget', 'shadow'):
            choice = choose(rows, self.budget, self.chunk, free, total, self.block_size,
                            self.max_num_running_reqs, self.max_model_len)
        if self.method == 'budget':
            self.max_num_scheduled_tokens = choice['budget']
            self.scheduler_config.long_prefill_token_threshold = choice['chunk']
        decision_ms = (time.perf_counter() - start) * 1000
        native_start = time.perf_counter()
        try:
            output = super().schedule()
        finally:
            self.max_num_scheduled_tokens = self.budget
            self.scheduler_config.long_prefill_token_threshold = self.chunk
        native_ms = (time.perf_counter() - native_start) * 1000
        # Common diagnostic accounting remains in end-to-end engine timing.
        audit_start = time.perf_counter()
        expected = (choice['selected'] if self.method == 'budget' else choice['base']) if choice else project(
            rows, self.budget, self.chunk, free, self.block_size,
            self.max_num_running_reqs, self.max_model_len)
        actual = output.num_scheduled_tokens
        after = manager.block_pool.get_num_free_blocks()
        preemptions = sum(r.running and self.requests[r.rid].status == RequestStatus.PREEMPTED for r in rows)
        if expected is not None:
            assert actual == expected.tokens, (actual, expected.tokens)
            assert free - after == expected.growth, (free, after, expected.growth)
            assert preemptions == 0
        recomputed = 0
        details = []
        for r in rows:
            computed = self.requests[r.rid].num_computed_tokens - actual[r.rid] if r.rid in actual else r.computed
            high = max(self.high_water.get(r.rid, 0), r.computed)
            n = actual.get(r.rid, 0)
            recomputed += min(n, max(0, high - computed))
            self.high_water[r.rid] = max(high, computed+n)
            if n:
                details.append(dict(id=r.rid, before=computed, tokens=n,
                                    prompt=r.prompt, outputs=r.outputs,
                                    blocks_before=r.blocks, blocks_after=len(tables.get(r.rid, ())),
                                    new_admission=not r.running))
        record = dict(round=len(self.round_rows), method=self.method, free_before=free,
                      free_after=after, total_blocks=total, preemptions=preemptions,
                      recomputed_tokens=recomputed, scheduled=dict(actual), allocations=details,
                      budget=choice['budget'] if self.method=='budget' else self.budget,
                      chunk=choice['chunk'] if self.method=='budget' else self.chunk,
                      proposed_budget=choice['budget'] if choice else self.budget,
                      candidates=choice['candidates'] if choice else 0,
                      proposed_change=choice['changed'] if choice else False,
                      intervened=bool(choice and choice['changed'] and self.method=='budget'),
                      recovery_fallback=expected is None, verified=expected is not None,
                      decision_ms=decision_ms, native_ms=native_ms)
        base = choice['base'] if choice else None
        record['baseline'] = base.tokens if base is not None else None
        record['baseline_growth_blocks'] = base.growth if base is not None else None
        record['reductions'] = [dict(id=r.rid, baseline_tokens=base.tokens[r.rid],
            actual_tokens=actual.get(r.rid, 0), computed=r.computed,
            remaining=r.remaining, outputs=r.outputs, running=r.running)
            for r in rows if base is not None and base.tokens.get(r.rid, 0)>actual.get(r.rid, 0)]
        record['audit_ms'] = (time.perf_counter()-audit_start)*1000
        record['schedule_ms'] = (time.perf_counter()-start)*1000
        self.round_rows.append(record)
        return output
