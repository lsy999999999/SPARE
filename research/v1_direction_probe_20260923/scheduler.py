"""Isolated experimental adapter; original scheduler files remain frozen.

The scoped selector patch is process-local and requires sequential episodes.
GPU parallelism must use separate processes, never threads sharing a scheduler.
"""
from unittest.mock import patch
from research.v1_state_aware_20260923 import scheduler as original
from research.v1_direction_probe_20260923.policy import METHODS, select


class ProbeScheduler(original.StateScheduler):
    def configure(self, method, budget, chunk, concurrency, **kwargs):
        self.probe_method = method
        self.last_probe = {}
        self.last_features = {}
        super().configure('pressure_one' if method in METHODS[3:] else method,
                          budget, chunk, concurrency, **kwargs)

    def _select(self, method, rows, *args):
        choice = select(self.probe_method, rows, *args)
        self.last_probe = choice.get('probe', {})
        self.last_features = dict(running=sum(r.running for r in rows),
            decode=sum(r.decode for r in rows),
            pending_first=sum(r.outputs == 0 for r in rows),
            known_prefill_remaining=sum(max(0, r.prompt-r.computed) for r in rows if r.outputs == 0))
        return choice

    def schedule(self):
        with patch.object(original, 'select', self._select):
            out = super().schedule()
        self.round_rows[-1].update(probe=self.last_probe, features=self.last_features,
                                  experiment_method=self.probe_method)
        return out
