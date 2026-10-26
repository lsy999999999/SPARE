import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.v1_fixed_order_20260927.common import *


def main():
    assert not (OUT/'design.json').exists(), 'Do not overwrite frozen design'
    cases = []
    for source in ('code', 'conversation'):
        for shape in ('steady', 'burst_stop'):
            case = f'{source}_w1_{shape}'
            original = ROOT/f'artifacts/v1_broad12h_20260925_r1/inputs/{case}.json'
            reqs = read(original)
            assert len(reqs) == 64 and all(0 < r['output'] <= 512 and r['prompt']+r['output'] <= 8000 for r in reqs)
            write(OUT/f'inputs/{case}.json', reqs)
            cases.append(dict(case=case, source=source, shape=shape, original=str(original.relative_to(ROOT))))
    groups = [dict(key=f'{source}_c{capacity}', source=source, capacity=capacity)
              for capacity in (768, 1536, 2048) for source in ('code', 'conversation')]
    frozen_sources = sources()
    for relative in frozen_sources:
        target = OUT/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
    write(OUT/'design.json', dict(frozen_at=time.time(), claim='exploratory reused-window action pilot',
          cases=cases, groups=groups, arms=ARMS, configs={a:dict(BASE,threshold=spec(a)) for a in ARMS},
          repeats=2, formal_expected=144, branch_maximum=168, timeout_s=180,
          source_hashes=frozen_sources, input_hashes={c['case']:digest(OUT/f"inputs/{c['case']}.json") for c in cases},
          protocol_sha256=digest(HERE/'PROTOCOL_CN.md')))
    print('Frozen 6 groups / 12 scenarios / 144 formal episodes; at most 168 branch episodes.')


if __name__ == '__main__':
    main()
