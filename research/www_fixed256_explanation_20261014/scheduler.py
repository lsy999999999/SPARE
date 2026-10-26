"""Keep the variant selector scoped to this dedicated worker process."""
from research.www_evaluation_gpu_completion_20261003 import scheduler as baseline
from research.www_fixed256_explanation_20261014.policy import select

baseline.select = select


class ExplanationScheduler(baseline.GPUBaselineScheduler):
    pass
