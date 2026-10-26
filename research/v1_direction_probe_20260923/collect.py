"""Assemble clean cells with provenance, retaining all interrupted originals."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'artifacts/v1_direction_probe_20260923_gpu_r1'
REC=ROOT/'artifacts/v1_direction_probe_20260923_gpu_recovery_r1'
OUT=ROOT/'artifacts/v1_direction_probe_20260923_gpu_final'


def read(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert read(REC/'status.json')['state']=='finished'
    assert read(OLD/'worker_g1.json')['state']=='finished'
    selected=[]; excluded=[]; provenance={}
    for folder in (OLD,REC):
        for p in sorted((folder/'runs').glob('*.json')):
            r=read(p)
            if folder==OLD and r['job']['cell']['capacity']==768:
                excluded.append(str(p.relative_to(ROOT)))
                continue
            assert r['status']=='complete'
            assert digest(folder/r['rounds_path'])==r['rounds_sha256']
            selected.append(r)
            for src,rel in ((p,Path('runs')/p.name),(folder/r['rounds_path'],Path(r['rounds_path']))):
                dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(src,dst)
            provenance[r['job']['key']]=dict(source=str(p.relative_to(ROOT)),sha256=digest(p))
    assert len(selected)==34 and sum(r['job']['phase']=='compare' for r in selected)==28
    m=dict(methods=read(OLD/'manifest.json')['methods'],compare_expected=28,aa_expected=6,
        source_manifests={str(p.relative_to(ROOT)):digest(p) for p in (OLD/'manifest.json',REC/'manifest.json')},
        provenance=provenance,excluded_interrupted_cell=excluded,
        interruption=read(OLD/'status.json'),
        design='2048: two reverse-order repeats on GPU1; 768: complete matched methods on each of GPU6/GPU7, opposite orders')
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(dict(rows=selected,manifest=m),indent=2)+'\n')
    print('Collected',len(selected),'clean runs; excluded',len(excluded),'interrupted-cell records')


if __name__=='__main__':main()
