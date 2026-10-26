"""Freeze a paired GPU fill-in for the missing sustained/fixed512 cell."""
import shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.www_fixed512_sustained_20261014.common import *

def main():
    assert not (OUT/'design.json').exists(), 'already frozen'
    prior=ROOT/'artifacts/www_capacity_load_20261007_r1'
    original=read(prior/'design.json')
    cells=[]
    for raw in original['cells']:
        if raw['family']!='load': continue
        c=dict(raw)
        for field in ('input','history_file'):
            source=prior/c[field]
            target=OUT/('inputs' if field=='input' else 'history')/source.name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target);c[field]=str(target.relative_to(OUT))
        assert digest(OUT/c['input'])==c['input_sha256']
        assert len(read(OUT/c['input']))==512
        cells.append(c)
    assert len(cells)==24
    # Preserve the archived dependencies used by the earlier GPU matrix.
    for logical,source in source_paths().items():
        archived=prior/'source'/logical
        use=archived if archived.exists() else source
        target=OUT/'source'/logical;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(use,target)
    design=dict(original)
    design.update(frozen_at=time.time(),cells=cells,methods=list(ARMS),options=OPTIONS,
        expected_formal_runs=192,expected_aa_runs=48,gpu_candidates=[2,6],
        deadline_hours=120,episode_timeout_s=1800,source_hashes=source_hashes(),
        provenance=dict(design=str((prior/'design.json').relative_to(ROOT)),sha256=digest(prior/'design.json')),
        objective='fill sustained-load fixed512 comparison with same-batch SPARE; paired 24 cells x 4 repeats')
    write(OUT/'design.json',design)
    print(json.dumps(dict(cells=24,formal_runs=192,aa_runs=48)))
if __name__=='__main__':main()
