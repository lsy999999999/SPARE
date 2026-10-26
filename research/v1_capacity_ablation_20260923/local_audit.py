"""Local CPU policy tests, input checks and historical request-ID overlap scan."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.v1_capacity_ablation_20260923.verify_inputs import main as verify


def main():
    started=time.time()
    verify()
    subprocess.run([sys.executable,'-B','-m','unittest',
        'research.v1_capacity_ablation_20260923.test_policy','-v'],cwd=ROOT,check=True)
    out=ROOT/'artifacts/v1_capacity_design_20260923'
    m=json.loads((out/'input_manifest.json').read_text())
    patterns=[]
    for block in m['blocks'].values():
        source=block['source']
        for row in block['rows']:
            index=row['source_row']
            patterns.append(f'"{source}-r{index}"')
            patterns.append(f'"source_row_index"[[:space:]]*:[[:space:]]*{index}([,}}[:space:]])')
    with tempfile.NamedTemporaryFile(mode='w',prefix='capacity-history-',suffix='.patterns') as f:
        f.write('\n'.join(patterns)+'\n'); f.flush()
        cmd=['rg','-l','-f',f.name,'--glob','*.json','--glob','*.jsonl',
             '--glob','!v1_capacity_design_20260923/**','--glob','!v1_capacity_gpu_20260923_r1/**',
             '--glob','!v1_capacity_ablation_20260923/**','artifacts','research']
        result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    matches=[line for line in result.stdout.splitlines() if '/v1_capacity_' not in line]
    assert result.returncode in (0,1), result.stderr
    report=dict(complete=not matches,policy_tests='passed',inputs='passed',
        historical_overlap_matches=matches,scope='Repository research/artifacts JSON/JSONL, canonical Azure request IDs and source_row_index; previous declared source windows checked by builder',
        limitations='Does not establish independence from unrecorded external experiments or ID-stripped data',
        elapsed_s=time.time()-started,input_manifest_sha256=hashlib.sha256((out/'input_manifest.json').read_bytes()).hexdigest(),
        measured_gpu_runs=0)
    (out/'local_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if matches: raise SystemExit('Historical overlap candidates require inspection before GPU comparison')


if __name__=='__main__': main()
