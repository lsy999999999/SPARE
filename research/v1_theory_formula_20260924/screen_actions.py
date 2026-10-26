"""Same-state action screening on old logs; never interprets it as counterfactual TTFT."""
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_theory_formula_20260924.common import ROOT,OUT,read,write,digest
from research.v1_theory_formula_20260924.policy import values,arms
from research.v1_saving_threshold_20260924.policy import threshold_value


def main():
    previous=ROOT/'artifacts/v1_constant_shift_20260924_r1'
    specs=arms();rows=[];checked=0;sources=[]
    for source in ('code','conversation'):
        for cap in (768,1536):
            counts={a:dict(eligible=0,different_old=0,different_decode=0,interventions=0) for a in specs}
            for repeat in (0,1):
                path=previous/'runs'/f'compare_{source}_fresh_c{cap}_old_r{repeat}.json'
                r=read(path);trace=previous/r['rounds_path'];assert digest(trace)==r['rounds_sha256']
                sources.append(dict(run=str(path.relative_to(ROOT)),run_sha256=digest(path),rounds_sha256=digest(trace)))
                with gzip.open(trace,'rt') as stream:rounds=json.load(stream)
                for row in rounds:
                    w=sum(max(0,s['prompt']-s['computed']) for s in row['state'] if s['outputs']==0)
                    held=sum(s['blocks'] for s in row['state'] if s['outputs']>0)
                    f=values(row['free_before'],w,held,cap)
                    ns={a:threshold_value(spec,f) for a,spec in specs.items()}
                    assert ns['old']==row['threshold'];checked+=1
                    if row['baseline'] is None:continue
                    cs=[c for c in row['candidate_diagnostics'] if c['tokens'] is not None and c['guard']]
                    def action(n):
                        if len(row['state'])<2 or row['baseline_growth_blocks']<=0:return row['baseline']
                        return next((c['tokens'] for c in cs if c['saving']>=n),row['baseline'])
                    actions={a:action(n) for a,n in ns.items()}
                    assert actions['old']==row['scheduled']
                    eligible=any(c['saving']>=1 for c in cs) and len(row['state'])>=2 and row['baseline_growth_blocks']>0
                    if not eligible:continue
                    for a in specs:
                        c=counts[a];c['eligible']+=1
                        c['different_old']+=actions[a]!=actions['old']
                        c['different_decode']+=actions[a]!=actions['decode']
                        c['interventions']+=actions[a]!=row['baseline']
            rows.extend(dict(source=source,capacity=cap,arm=a,**c) for a,c in counts.items())
    write(OUT/'action_screen.json',dict(old_equivalence_rounds=checked,rows=rows,sources=sources,
        scope='Eight existing old-formula trajectories, promoted to exploratory development. Same-state actions only; no new-policy TTFT labels or closed-loop inference.'))
    lines=['# 理论候选在旧轨迹上是否真的改变动作','',
           f'核对{checked:,}轮旧式门槛/动作一致性，8条原始轨迹哈希通过。以下合并两次重复。只检查动作区别，不表示新公式的TTFT。','',
           '| 来源 | 容量 | 公式 | 有正节约候选的轮数 | 与同状态旧式不同 | 与同状态旧+D不同 | 干预轮数 |',
           '|---|---:|---|---:|---:|---:|---:|']
    for row in rows:lines.append('| {source} | {capacity} | {arm} | {eligible} | {different_old} | {different_decode} | {interventions} |'.format(**row))
    (ROOT/'research/v1_theory_formula_20260924/ACTION_SCREEN_CN.md').write_text('\n'.join(lines)+'\n')
    print('PASS:',checked,'old-equivalence rounds; action screening only.')


if __name__=='__main__':main()
