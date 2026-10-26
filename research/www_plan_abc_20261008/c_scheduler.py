"""Isolated G2 scheduler with the guarded fixed candidate added."""
from research.www_evaluation_gpu_completion_20261003 import scheduler as g2_scheduler
from research.www_plan_abc_20261008.c_policy import select

g2_scheduler.select = select


class CScheduler(g2_scheduler.GPUBaselineScheduler):
    pass
