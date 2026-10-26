"""Dispatch to unchanged audited rules; no new online scheduling policy."""
from research.v1_state_aware_20260923.scheduler import StateScheduler
from research.v1_paper_comparison_20260922.scheduler import ComparisonScheduler, PAPER_METHODS


class PilotScheduler(StateScheduler, ComparisonScheduler):
    # MRO: State -> Comparison -> Budget -> native. Comparison's super calls
    # reach Budget; State.schedule explicitly calls the native scheduler.
    def configure(self, method, budget, chunk, concurrency, **kwargs):
        if method in PAPER_METHODS:
            assert not kwargs
            ComparisonScheduler.configure(self, method, budget, chunk, concurrency)
        else:
            StateScheduler.configure(self, method, budget, chunk, concurrency, **kwargs)

    def schedule(self):
        if self.method in PAPER_METHODS:
            return ComparisonScheduler.schedule(self)
        return StateScheduler.schedule(self)
