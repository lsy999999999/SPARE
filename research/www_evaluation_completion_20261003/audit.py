"""Independent post-run arithmetic and file-integrity audit; no policy changes."""
import csv
import gzip
import json
import math
from collections import Counter
from .common import OUT, ROOT, read, write, digest


def audit():
    design=read(OUT/'design.json');counts=Counter();failures=[]
    for rel,sha in design['source_hashes'].items():
        assert digest(ROOT/rel)==sha,rel
    inputs={}
    for task in design['tasks']:
        try:
            folder=OUT/'runs'/task['name'];result=read(folder/'result.json')
            assert result['complete'] and result['task']==task
            path=ROOT/task['input']
            if str(path) not in inputs:inputs[str(path)]=(read(path),digest(path))
            reqs,sha=inputs[str(path)];assert result['input_sha256']==sha
            requested={r['id']:r for r in reqs}
            actual=list(csv.DictReader((folder/'requests.csv').open()))
            assert len(actual)==len(reqs)==len({r['request_id'] for r in actual})
            assert {r['request_id'] for r in actual}==set(requested)
            ttft=[];events=[]
            for r in actual:
                spec=requested[r['request_id']]
                a=float(r['arrival_ms']);s=float(r['submitted_ms']);f=float(r['first_token_ms']);e=float(r['finish_ms'])
                assert a==spec['arrival_ms'] and 0<=a<=s<=f<=e
                assert int(r['prompt_tokens'])==spec['prompt'] and int(r['output_tokens'])==spec['output']
                assert math.isclose(f-a,float(r['ttft_ms']),abs_tol=1e-7)
                ttft.append(f-a);events.extend([(a,1),(f,-1)])
            events.sort();area=0.;last=0.;pending=0
            for t,change in events:
                area+=pending*(t-last);pending+=change;last=t
                assert pending>=0
            m=result['metrics']
            assert pending==0 and math.isclose(area,sum(ttft),rel_tol=1e-10,abs_tol=1e-6)
            assert math.isclose(sum(ttft)/len(ttft),m['ttft_ms_mean'],rel_tol=1e-10)
            assert m['executed_tokens']-m['recomputed_tokens']==sum(r['prompt']+r['output']-1 for r in reqs)
            count=0;last_end=0.;preemptions=recomputed=0
            with gzip.open(folder/'iterations.jsonl.gz','rt') as h:
                for line in h:
                    r=json.loads(line);count+=1
                    assert 0<=r['free_before']<=task['capacity']-1
                    assert 0<=r['free_after']<=task['capacity']-1
                    assert r['simulated_start_ms']>=last_end
                    assert r['simulated_finish_ms']>r['simulated_start_ms']
                    assert math.isclose(r['simulated_finish_ms']-r['simulated_start_ms'],r['simulated_step_ms'],rel_tol=1e-8,abs_tol=1e-7)
                    last_end=r['simulated_finish_ms'];preemptions+=r['preemptions'];recomputed+=r['recomputed_tokens']
            assert count==m['rounds'] and preemptions==m['preemptions'] and recomputed==m['recomputed_tokens']
            if task['arm'] in ('spare','arrival','basic'):
                assert m.get('finish_opportunities_deferred',0)==0
                assert m.get('normal_decode_reductions',0)==0
            counts['trajectories']+=1;counts['iterations']+=count;counts['requests']+=len(reqs)
        except Exception as e:
            failures.append(dict(name=task['name'],error=repr(e)))
    report=dict(passed=not failures and counts['trajectories']==design['expected'],
                audit_source_sha256=digest(__file__),design_sha256=digest(OUT/'design.json'),
                counts=dict(counts),failures=failures,
                scope='Independent input/output/timing/queue-area/block-bound/iteration/recompute/source audit. Not proof of GPU prediction accuracy.')
    write(OUT/'audit.json',report);print(json.dumps(report,ensure_ascii=False))
    if failures:raise RuntimeError('Independent audit failed')

if __name__=='__main__':audit()
