"""Read-only progress, including paired results only when both repeats exist."""
from collections import defaultdict
import json
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.v1_capacity_paper_pilot_20260923.common import *

rows = [read(p) for p in sorted((OUT/'runs').glob('*.json'))]
groups = defaultdict(list)
for r in rows:
    if r['job']['phase'] == 'compare' and r['status'] == 'complete':
        groups[(r['job']['cell']['key'], r['job']['arm'])].append(r['metrics']['mean_ttft_ms'])
paired = {f'{c}/{a}': round(st.fmean(v),1) for (c,a),v in groups.items() if len(v)==2}
print(json.dumps(dict(state=read(OUT/'status.json')['state'], saved=len(rows),
    compare=sum(r['job']['phase']=='compare' for r in rows),
    failures=[(r['job']['key'],r['error']) for r in rows if r['status']!='complete'],
    workers=[read(p) for p in sorted(OUT.glob('worker_g*.json'))],
    cells=[read(p) for p in sorted((OUT/'cells').glob('*.json'))], paired_ttft_ms=paired),indent=2))
