"""Published scheduling rules adapted to the same native V1 engine.

The frozen A/SRP execution path is inherited without a policy change.
These are paper-rule adaptations, not the Sarathi/LightLLM serving systems.
"""
import hashlib
import math
import time

from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.request import RequestStatus

from research.v1_budget_20260922.scheduler import BudgetScheduler
from research.v1_budget_20260922.policy import State, project
from research.v1_paper_comparison_20260922.rules import (
    HistoryPredictor, apt_kv_density, censored_log_t_score,
    future_peak_blocks, uniboost_priority)

PAPER_METHODS = (
    'sarathi_v1', 'past_future_v1', 'ssjf_oracle', 'mc_sf_oracle',
    'trail_oracle', 'tie_oracle_proxy', 'uniboost_static',
    'apt_serve_kv_rule', 'sola_rule')
ORACLE_METHODS = ('ssjf_oracle', 'mc_sf_oracle', 'trail_oracle',
                  'tie_oracle_proxy')


class ComparisonScheduler(BudgetScheduler):
    manual_priority_preemption = True

    def configure(self, method, budget, chunk, concurrency):
        super().configure('native' if method in PAPER_METHODS else method,
                          budget, chunk, concurrency)
        self.method = method
        self._held = []
        self.predictor = HistoryPredictor(getattr(self, 'history_context', ()),
                                          getattr(self, 'seed_context', 0))
        self.oracle_lengths = dict(getattr(self, 'oracle_lengths_context', {}))
        if method in ORACLE_METHODS and not self.oracle_lengths:
            raise ValueError('oracle baseline requires an explicit realized-length map')
        self.paper_round = {}
        self._manual_preemptions = 0
        self._last_service = {}
        self._tie_base_scores = {}

    def _free_request(self, request):
        if (getattr(self, 'method', None) == 'past_future_v1'
                and request.status in (RequestStatus.FINISHED_STOPPED,
                                       RequestStatus.FINISHED_LENGTH_CAPPED)):
            self.predictor.observe(len(request.output_token_ids))
        return super()._free_request(request)

    def _order(self):
        if self.method not in PAPER_METHODS:
            return super()._order()
        self.paper_round = dict(gate_checked=0, gate_rejected=0,
                                forecast_peak_blocks=0, held_count=0)
        if self.method == 'ssjf_oracle':
            # The SSJF paper uses a non-preemptive shortest-predicted-job-first
            # waiting queue. This diagnostic substitutes realized lengths.
            self.waiting = FCFSRequestQueue(sorted(
                self.waiting,
                key=lambda r: (self.oracle_lengths[r.request_id],
                               r.arrival_time, r.request_id)))
            return

        if self.method == 'trail_oracle':
            self._trail_order()
            return

        if self.method == 'tie_oracle_proxy':
            self._tie_order()
            return

        if self.method == 'uniboost_static':
            self._uniboost_order()
            return

        if self.method == 'apt_serve_kv_rule':
            self._apt_order()
            return

        if self.method == 'sola_rule':
            self._sola_order()
            return

        # Sarathi Algorithm 3: ongoing decodes, partial prefills, new FIFO.
        # MC-SF also retains in-progress requests. V1 keeps its incremental
        # block allocation and recovery for every paper-rule adaptation.
        decodes = [r for r in self.running if len(r.output_token_ids) > 0
                   and r.num_tokens - r.num_computed_tokens == 1]
        decode_ids = {r.request_id for r in decodes}
        self.running[:] = decodes + [r for r in self.running
                                     if r.request_id not in decode_ids]
        if self.method == 'sarathi_v1' or not self.waiting:
            return
        manager = self.kv_cache_manager
        tables = manager.coordinator.single_type_managers[0].req_to_blocks
        free = manager.block_pool.get_num_free_blocks()
        running_rows = [State(r.request_id, r.num_computed_tokens, r.num_tokens,
                              r.num_prompt_tokens, len(r.output_token_ids),
                              len(tables.get(r.request_id, ())), True)
                        for r in self.running]
        plan = project(running_rows, self.budget, self.chunk, free,
                       self.block_size, self.max_num_running_reqs, self.max_model_len)
        # Native recovery schedules no waiting requests in a preempting round.
        if plan is None:
            return
        budget = self.budget - sum(plan.tokens.values())
        slots = self.max_num_running_reqs - len(self.running)
        if budget <= 0 or slots <= 0:
            return
        states = []
        for r in self.running:
            generated = len(r.output_token_ids)
            if self.method == 'mc_sf_oracle':
                prediction = self.oracle_lengths[r.request_id]
            else:
                # Do not read r.max_tokens: replay uses a common context cap.
                cap = self.max_model_len - r.num_prompt_tokens
                prediction = self.predictor.sample(generated, cap)
            states.append((r.num_prompt_tokens, generated, prediction))
        allowed = []
        waiting = list(self.waiting)
        if self.method == 'mc_sf_oracle':
            waiting.sort(key=lambda r: (self.oracle_lengths[r.request_id],
                                        r.arrival_time, r.request_id))
        remaining_free = free - plan.growth
        for r in waiting:
            if budget <= 0 or slots <= 0:
                break
            n = min(r.num_tokens - r.num_computed_tokens, budget, self.chunk)
            if n <= 0:
                break
            extra = max(0, (r.num_computed_tokens+n+15)//16
                        - len(tables.get(r.request_id, ())))
            if extra > remaining_free:
                break
            generated = len(r.output_token_ids)
            prediction = (self.oracle_lengths[r.request_id]
                          if self.method == 'mc_sf_oracle' else
                          self.predictor.sample(
                              generated, self.max_model_len-r.num_prompt_tokens))
            proposed = states + [(r.num_prompt_tokens, generated, prediction)]
            peak = future_peak_blocks(proposed, self.block_size)
            self.paper_round['gate_checked'] += 1
            self.paper_round['forecast_peak_blocks'] = max(
                self.paper_round['forecast_peak_blocks'], peak)
            # V1 reserves one null block, unavailable to requests.
            if peak > self.kv_cache_config.num_blocks - 1:
                self.paper_round['gate_rejected'] += 1
                break
            allowed.append(r)
            states = proposed
            budget -= n
            remaining_free -= extra
            slots -= 1
        self._held = waiting[len(allowed):]
        self.waiting = FCFSRequestQueue(allowed)
        self.paper_round['held_count'] = len(self._held)

    @staticmethod
    def _stable_seed(request_id):
        return int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8], 'big')

    def _preempt_for_priority(self, key, can_preempt=lambda request: True):
        """Apply the simulator victim rule when the backend supports it."""
        if self.manual_priority_preemption and self.waiting:
            candidates = [r for r in self.running if can_preempt(r)]
            if candidates:
                best_waiting = min(self.waiting, key=key)
                worst_running = max(candidates, key=key)
                if key(best_waiting) < key(worst_running):
                    self.running.remove(worst_running)
                    self.kv_cache_manager.free(worst_running)
                    worst_running.status = RequestStatus.PREEMPTED
                    worst_running.num_computed_tokens = 0
                    self.waiting.append(worst_running)
                    self._manual_preemptions += 1
        self.running.sort(key=key)
        self.waiting = FCFSRequestQueue(sorted(self.waiting, key=key))

    def _trail_order(self):
        def key(request):
            remaining = max(1, self.oracle_lengths[request.request_id]
                            - len(request.output_token_ids))
            return remaining, request.arrival_time, request.request_id

        def early(request):
            predicted = self.oracle_lengths[request.request_id]
            return len(request.output_token_ids) < math.floor(.8 * predicted)
        self._preempt_for_priority(key, early)
        self.paper_round.update(trail_c=.8,
            trail_priority_only=int(not self.manual_priority_preemption),
            trail_preemptions=self._manual_preemptions)

    def _tie_order(self):
        queue_length = len(self.waiting)
        beta = min(.5, max(.1, .1 * queue_length / self.max_num_running_reqs))
        now = getattr(self, 'arrival_clock', time.monotonic)()

        def key(request):
            rid = request.request_id
            cache_key = (rid, beta)
            if cache_key not in self._tie_base_scores:
                realized = self.oracle_lengths[rid]
                cap = self.max_model_len - request.num_prompt_tokens
                self._tie_base_scores[cache_key] = censored_log_t_score(
                    math.log(realized), .2, cap, beta,
                    seed=self._stable_seed(rid))
            wait_s = max(0., now - request.arrival_time)
            score = self._tie_base_scores[cache_key] * .9 ** (wait_s / 30.)
            return score, request.arrival_time, rid
        self.waiting = FCFSRequestQueue(sorted(self.waiting, key=key))
        self.paper_round.update(tie_beta=beta, tie_alpha=.9, tie_sigma_proxy=.2)

    def _uniboost_order(self):
        def key(request):
            signal = (request.num_prompt_tokens if not request.output_token_ids
                      else len(request.output_token_ids))
            return (uniboost_priority(request.arrival_time, signal),
                    request.arrival_time, request.request_id)
        self._preempt_for_priority(key)
        self.paper_round.update(uniboost_gamma=1e-4, uniboost_bin_size=16,
            uniboost_priority_only=int(not self.manual_priority_preemption),
            uniboost_preemptions=self._manual_preemptions)

    def _pending(self, request, now):
        return max(0., now - self._last_service.get(
            request.request_id, request.arrival_time))

    def _apt_order(self):
        now = getattr(self, 'arrival_clock', time.monotonic)()
        tables = self.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
        waiting_pending = sum(self._pending(r, now) for r in self.waiting)
        running_pending = sum(self._pending(r, now) for r in self.running)
        phase = 'prefill' if waiting_pending > running_pending else 'decode'

        def key(request):
            total_blocks = math.ceil((request.num_prompt_tokens +
                len(request.output_token_ids) + 1) / self.block_size)
            resident = len(tables.get(request.request_id, ()))
            required = max(1, total_blocks - resident)
            density = apt_kv_density(self._pending(request, now), required)
            return -density, request.arrival_time, request.request_id
        self.running.sort(key=key)
        self.waiting = FCFSRequestQueue(sorted(self.waiting, key=key))
        self.paper_round.update(apt_phase=phase,
            apt_waiting_pending_s=waiting_pending,
            apt_running_pending_s=running_pending)

    def _sola_order(self):
        now = getattr(self, 'arrival_clock', time.monotonic)()
        def prefill_risk(request):
            remaining = max(1, request.num_prompt_tokens -
                            request.num_computed_tokens)
            context = request.num_computed_tokens
            predicted_s = (6. + .08 * remaining +
                           .000003 * context * remaining) / 1000
            return (max(0., now-request.arrival_time) + predicted_s) / 1.0

        def decode_risk(request):
            context = request.num_prompt_tokens + len(request.output_token_ids)
            predicted_s = (6.03 + .00001 * context) / 1000
            return (self._pending(request, now) + predicted_s) / .2

        waiting_pressure = max(map(prefill_risk, self.waiting), default=0.)
        decode_pressure = max(map(decode_risk, self.running), default=0.)
        phase = 'prefill' if waiting_pressure >= decode_pressure else 'decode'
        self.waiting = FCFSRequestQueue(sorted(
            self.waiting,
            key=lambda r: (-prefill_risk(r),
                           r.arrival_time, r.request_id)))
        self.running.sort(key=lambda r: (-decode_risk(r),
                                         r.arrival_time, r.request_id))
        self.paper_round.update(sola_phase=phase,
            sola_ttft_pressure=waiting_pressure,
            sola_tpot_pressure=decode_pressure,
            sola_ttft_slo_ms=1000, sola_tpot_slo_ms=200)

    def schedule(self):
        if self.method not in PAPER_METHODS:
            return super().schedule()
        start = time.perf_counter()
        try:
            output = super().schedule()
        finally:
            # Native may have prepended newly preempted requests. Preserve them.
            self.waiting.extend(self._held)
            self._held = []
        row = self.round_rows[-1]
        if self._manual_preemptions:
            row['preemptions'] += self._manual_preemptions
        now = getattr(self, 'arrival_clock', time.monotonic)()
        for rid in output.num_scheduled_tokens:
            self._last_service[rid] = now
        row.update(self.paper_round)
        row['schedule_ms'] = (time.perf_counter()-start)*1000
        self._manual_preemptions = 0
        return output
