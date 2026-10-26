"""Independent integrity check of saved records, without claiming complete comparisons."""
import json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_broad12h_20260925.common import OUT,read,write
from research.v1_broad12h_20260925.analysis.report import audit
paths=sorted((OUT/'attempts').glob('*/a*/runs/*.json'))
count=0;rounds=0
for p in paths:
    r=read(p)
    if r['status']!='complete':continue
    a=audit(r);rounds+=a['rounds'];count+=1
result=dict(passed=True,records=count,rounds=rounds,updated=time.time(),scope='Saved successful attempt records, not necessarily accepted full groups; no performance conclusion.')
write(OUT/'analysis/live_audit.json',result)
print(json.dumps(result))
