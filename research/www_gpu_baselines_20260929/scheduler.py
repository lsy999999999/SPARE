"""Exact GPU dispatch for the 13 policies used by writing/www/code/run_simulator.py."""
import time

from research.v1_broad12h_20260925.scheduler import FormulaScheduler
from research.v1_budget_20260922.scheduler import BudgetScheduler
from research.v1_paper_comparison_20260922.scheduler import (
    ComparisonScheduler, PAPER_METHODS,
)


class GPUBaselineScheduler(FormulaScheduler, ComparisonScheduler):
    # Manual victim replacement does not emit V1's required preemption output.
    manual_priority_preemption = False

    def configure(self, method, budget, chunk, concurrency, **kwargs):
        if method == "threshold":
            FormulaScheduler.configure(self, method, budget, chunk,
                                       concurrency, **kwargs)
        else:
            if kwargs:
                raise ValueError(f"unexpected policy arguments: {kwargs}")
            ComparisonScheduler.configure(self, method, budget, chunk,
                                          concurrency)

    def add_request(self, request):
        if self.method == "threshold":
            return FormulaScheduler.add_request(self, request)
        return BudgetScheduler.add_request(self, request)

    def schedule(self):
        if self.method == "threshold":
            return FormulaScheduler.schedule(self)
        start = time.perf_counter()
        tables = self.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
        state = [dict(
            rid=request.request_id,
            computed=request.num_computed_tokens,
            tokens=request.num_tokens,
            prompt=request.num_prompt_tokens,
            outputs=len(request.output_token_ids),
            blocks=len(tables.get(request.request_id, ())),
            running=request in self.running,
        ) for request in list(self.running) + list(self.waiting)]
        output = (ComparisonScheduler.schedule(self)
                  if self.method in PAPER_METHODS
                  else BudgetScheduler.schedule(self))
        self.round_rows[-1]["state"] = state
        self.round_rows[-1]["schedule_ms"] = (time.perf_counter() - start) * 1000
        return output
