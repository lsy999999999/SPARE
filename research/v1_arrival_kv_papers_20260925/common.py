"""Frozen A-only capacity comparison on the existing V1 execution backend."""
from pathlib import Path
from research.v1_saving_threshold_20260924.common import read,write,digest,object_digest,BASE,MODEL
from research.v1_broad12h_20260925.policy import formula
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v1_arrival_kv_papers_20260925_r1'
CAPS=[512,640,768,896,1024,1280,1536,1792,2048,2560,3072,4096]
ARMS={
 'old':dict(method='threshold',config=dict(BASE,threshold=formula(0))),
 'arrival':dict(method='threshold',config=dict(BASE,threshold=formula(1))),
 'srp':dict(method='srp',config=BASE),
 'native':dict(method='native',config=BASE),
 'sarathi_common':dict(method='sarathi_v1',config=BASE),
 'past_future_common':dict(method='past_future_v1',config=BASE),
 'sarathi_frozen':dict(method='sarathi_v1',config=dict(BASE,chunk=1024)),
 'past_future_frozen':dict(method='past_future_v1',config=dict(BASE,chunk=1024)),
}

def source_hashes():
 files=list(HERE.glob('*.py'))
 for folder,names in (
 ('v1_broad12h_20260925',('policy.py','scheduler.py','common.py','analysis/report.py')),
 ('v1_saving_threshold_20260924',('policy.py','common.py')),
 ('v1_budget_20260922',('policy.py','scheduler.py','run.py','test_scheduler.py')),
 ('v1_state_aware_20260923',('run.py','common.py')),
 ('v1_paper_comparison_20260922',('scheduler.py','rules.py'))):
  files += [ROOT/'research'/folder/n for n in names]
 return {str(p.relative_to(ROOT)):digest(p) for p in files}

def context(scheduler,source,repeat):
 scheduler.history_context=read(OUT/'histories.json')[source]['lengths']
 scheduler.seed_context=2026092500+repeat
