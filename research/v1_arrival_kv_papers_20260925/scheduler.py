"""Dispatch unchanged formula and paper policies; native/SRP are instrumented controls."""
import time
from research.v1_broad12h_20260925.scheduler import FormulaScheduler
from research.v1_paper_comparison_20260922.scheduler import ComparisonScheduler,PAPER_METHODS
from research.v1_budget_20260922.scheduler import BudgetScheduler

class MatrixScheduler(FormulaScheduler,ComparisonScheduler):
 def configure(self,method,budget,chunk,concurrency,**kwargs):
  if method=='threshold':
   FormulaScheduler.configure(self,method,budget,chunk,concurrency,**kwargs)
  else:
   assert not kwargs
   ComparisonScheduler.configure(self,method,budget,chunk,concurrency)
 def add_request(self,request):
  if self.method=='threshold':return FormulaScheduler.add_request(self,request)
  return BudgetScheduler.add_request(self,request)
 def schedule(self):
  if self.method=='threshold':return FormulaScheduler.schedule(self)
  start=time.perf_counter()
  tables=self.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
  state=[dict(rid=r.request_id,computed=r.num_computed_tokens,tokens=r.num_tokens,
              prompt=r.num_prompt_tokens,outputs=len(r.output_token_ids),
              blocks=len(tables.get(r.request_id,())),running=r in self.running)
         for r in list(self.running)+list(self.waiting)]
  out=ComparisonScheduler.schedule(self) if self.method in PAPER_METHODS else BudgetScheduler.schedule(self)
  self.round_rows[-1]['state']=state
  self.round_rows[-1]['schedule_ms']=(time.perf_counter()-start)*1000
  return out
