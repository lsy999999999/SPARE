"""One GPU scheduler class for the frozen SPARE and S3 proxy adapters."""

from research.www_evaluation_gpu_completion_20261003.scheduler import (
    GPUBaselineScheduler as PaperScheduler,
)
from research.www_legacy_baselines_20261006.scheduler import (
    GPUBaselineScheduler as LegacyScheduler,
)
from vllm.v1.core.sched.scheduler import Scheduler as NativeScheduler


class IdeaScheduler(PaperScheduler, LegacyScheduler):
    def add_request(self, request):
        NativeScheduler.add_request(self, request)
        if self.method == "threshold":
            rid = request.request_id
            assert rid not in self.arrival_ids
            self.arrival_ids.add(rid)
            elapsed = self.arrival_clock() - self.arrival_origin
            self.arrival_estimator.observe(elapsed, request.num_prompt_tokens)
            self.arrival_pending.append(
                dict(id=rid, time_s=elapsed, prompt=request.num_prompt_tokens)
            )

    def configure(self, method, budget, chunk, concurrency, **kwargs):
        if method == "s3_proxy":
            return LegacyScheduler.configure(
                self, method, budget, chunk, concurrency, **kwargs
            )
        return PaperScheduler.configure(
            self, method, budget, chunk, concurrency, **kwargs
        )

    def schedule(self):
        if self.method == "s3_proxy":
            return LegacyScheduler.schedule(self)
        return PaperScheduler.schedule(self)
