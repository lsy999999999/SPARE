"""Same-state, one-action probes; all arrivals at zero, replay prefix EWMA times."""
import time
from research.v1_dynamic_actions_20260927.scheduler import DynamicScheduler


class ProbeScheduler(DynamicScheduler):
    def configure(self, method, budget, chunk, concurrency, **kwargs):
        self.probe_enabled=getattr(self,'probe_enabled_context',False)
        if not self.probe_enabled:
            self.arrival_clock=time.perf_counter
            return super().configure(method,budget,chunk,concurrency,**kwargs)
        self.probe_times=getattr(self,'probe_times_context',None)
        self.probe_target=getattr(self,'probe_target_context',None)
        self.probe_wall_origin=time.perf_counter()
        self.probe_virtual=0.
        self.probe_branch_wall=None
        self.arrival_clock=lambda:self.probe_virtual
        super().configure(method,budget,chunk,concurrency,**kwargs)

    def add_request(self,request):
        if not self.probe_enabled:
            return super().add_request(request)
        assert not self.round_rows,'Probes require all requests before first step'
        self.probe_virtual=0.
        return super().add_request(request)

    def schedule(self):
        if not self.probe_enabled:
            return super().schedule()
        index=len(self.round_rows);wall=time.perf_counter()
        if self.probe_times is not None and index<=self.probe_target:
            self.probe_virtual=self.probe_times[index]
        elif self.probe_branch_wall is not None:
            self.probe_virtual=self.probe_times[self.probe_target]+wall-self.probe_branch_wall
        else:
            self.probe_virtual=wall-self.probe_wall_origin
        if self.probe_target is not None and index==self.probe_target:
            self.probe_branch_wall=wall
        output=super().schedule()
        self.round_rows[-1]['probe_decision_wall']=wall
        return output
