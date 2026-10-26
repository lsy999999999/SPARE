"""G2 scheduler with a replayed logical clock and an auditable branch state."""
import hashlib
import json
import time

from research.www_evaluation_gpu_completion_20261003 import scheduler as g2_scheduler
from research.www_plan_abc_20261008.branch_policy import select

# The inherited schedule method resolves this module-level selector in its
# defining G2 module. Each GPU worker is a dedicated process for this probe.
g2_scheduler.select = select


class ProbeScheduler(g2_scheduler.GPUBaselineScheduler):
    def configure(self, method, budget, chunk, concurrency, **kwargs):
        self.probe_times = getattr(self, "probe_times_context", None)
        self.probe_target = getattr(self, "probe_target_context", None)
        self.probe_wall_origin = time.perf_counter()
        self.probe_virtual = 0.0
        self.probe_branch_wall = None
        self.arrival_clock = lambda: self.probe_virtual
        super().configure(method, budget, chunk, concurrency, **kwargs)

    def schedule(self):
        index = len(self.round_rows)
        wall = time.perf_counter()
        if self.probe_times is not None and index <= self.probe_target:
            self.probe_virtual = self.probe_times[index]
        elif self.probe_branch_wall is not None:
            self.probe_virtual = self.probe_times[self.probe_target] + wall - self.probe_branch_wall
        else:
            self.probe_virtual = wall - self.probe_wall_origin
        if self.probe_target is not None and index == self.probe_target:
            self.probe_branch_wall = wall

        manager = self.kv_cache_manager
        tables = manager.coordinator.single_type_managers[0].req_to_blocks
        state = dict(
            running=[r.request_id for r in self.running],
            waiting=[r.request_id for r in self.waiting],
            blocks={rid: len(blocks)
                    for rid, blocks in sorted(tables.items())},
            free=manager.block_pool.get_num_free_blocks(),
        )
        block_ids = {rid: [block.block_id for block in blocks]
                     for rid, blocks in sorted(tables.items())}
        physical_sha = hashlib.sha256(json.dumps(state, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        ids_sha = hashlib.sha256(json.dumps(block_ids, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        output = super().schedule()
        self.round_rows[-1]["probe_decision_wall"] = wall
        self.round_rows[-1]["physical_state_sha256"] = physical_sha
        self.round_rows[-1]["block_ids_sha256"] = ids_sha
        return output
