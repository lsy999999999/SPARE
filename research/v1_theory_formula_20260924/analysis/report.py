"""Independent reconstruction, paired comparisons and automatic Chinese report."""
import csv
import gzip
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_theory_formula_20260924.common import *
from research.v1_budget_20260922 import run as ref
ref.OUT=OUT


def reconstruct_threshold(spec,row):
    if spec=='off':return None
    if not isinstance(spec,dict):return int(spec)
    work=sum(max(0,s['prompt']-s['computed']) for s in row['state'] if s['outputs']==0)
    held=sum(s['blocks'] for s in row['state'] if s['outputs']>0)
    q=1+work/16; x=row['free_before']/q; u=1-held/(row['total_blocks']-1)
    if spec['features']==['coupled_exact_threshold']:
        raw=(q/16+row['free_before'])*math.expm1((8+32*u)/(q/16))
    elif spec['features']==['coupled_decode_threshold']:
        raw=(8+32*u)*(1+16*x)
    else:
        fs=dict(work_free_ratio=x,decoded_kv_headroom=u)
        raw=spec['intercept']+sum(c*fs[n] for n,c in zip(spec['features'],spec['coefficients']))
    return max(1,min(100000,math.ceil(raw)))


def audit(r):
    ref.audit_result(r)
    assert math.isclose(st.fmean(c['finish_ms']-c['arrival_ms'] for c in r['clocks'].values()),r['metrics']['mean_e2e_ms'],abs_tol=1e-6)
    with gzip.open(OUT/r['rounds_path'],'rt') as f:rounds=json.load(f)
    normal=0;diff=0;eligible=0;blocked=0;xs=[];zs=[];ns=[];mixed_tokens=[];mixed_ms=[]
    for row in rounds:
        state={s['rid']:s for s in row['state']}
        prefill=sum(n for rid,n in row['scheduled'].items() if state[rid]['outputs']==0)
        decode=sum(1 for rid in row['scheduled'] if state[rid]['outputs']>0 and state[rid]['tokens']-state[rid]['computed']==1)
        if prefill and decode:mixed_tokens.append(prefill);mixed_ms.append(row['step_ms'])
        work=sum(max(0,s['prompt']-s['computed']) for s in state.values() if s['outputs']==0)
        held=sum(s['blocks'] for s in state.values() if s['outputs']>0)
        x=row['free_before']/(1+work/16);z=held/(row['total_blocks']-1)
        assert 0<=z<=1
        for name,value in [('work_free_ratio',x),('decoded_kv_fraction',z),('decoded_kv_headroom',1-z),('decoded_kv_blocks',held)]:
            assert math.isclose(row['features'][name],value,abs_tol=1e-12)
        spec=r['job']['config']['threshold']
        n=reconstruct_threshold(spec,row)
        assert n==row['threshold']
        if row['baseline'] is None:continue
        normal+=1
        def growth(tokens):return sum(max(0,(state[rid]['computed']+count+15)//16-state[rid]['blocks']) for rid,count in tokens.items())
        assert growth(row['baseline'])==row['baseline_growth_blocks']
        assert growth(row['scheduled'])==row['free_before']-row['free_after']
        protected={rid:count for rid,count in row['baseline'].items() if
            (state[rid]['outputs']>0 and state[rid]['tokens']-state[rid]['computed']==1)
            or (state[rid]['outputs']==0 and count==state[rid]['tokens']-state[rid]['computed'])}
        assert all(row['scheduled'].get(rid,0)>=count for rid,count in protected.items())
        choices=[]
        for c in row['candidate_diagnostics']:
            if c['tokens'] is None:continue
            assert c['growth']==growth(c['tokens'])
            assert c['saving']==growth(row['baseline'])-c['growth']
            assert c['guard']==all(c['tokens'].get(rid,0)>=count for rid,count in protected.items())
            if c['guard']:choices.append(c)
        def action(n):
            if n is not None and len(state)>=2 and row['baseline_growth_blocks']>0:
                for c in choices:
                    if c['saving']>=n:return c['tokens']
            return row['baseline']
        assert row['scheduled']==action(n)
        old=max(1,min(100000,math.ceil(8+128*x)))
        diff+=action(old)!=row['scheduled']
        if action(1)!=row['baseline']:
            eligible+=1;blocked+=row['scheduled']==row['baseline'];xs.append(x);zs.append(z)
            if n is not None:ns.append(n)
    return dict(rounds=len(rounds),normal_rounds=normal,different_from_old=diff,
        mixed_rounds=len(mixed_tokens),mixed_prefill_tokens=st.fmean(mixed_tokens) if mixed_tokens else 0,
        mixed_step_ms=st.fmean(mixed_ms) if mixed_ms else 0,
        eligible_rounds=eligible,blocked_rounds=blocked,
        eligible_x_median=st.median(xs) if xs else None,eligible_z_median=st.median(zs) if zs else None,
        eligible_n_median=st.median(ns) if ns else None)


def main():
    assert read(OUT/'status.json')['state']=='complete'
    assert not list((OUT/'contamination').glob('*.json'))
    manifest=read(OUT/'manifest.json');design=read(OUT/'design.json');selection=read(OUT/'selection.json')
    assert digest(OUT/'design.json')==manifest['design_sha256']
    for name,h in manifest['sources'].items():assert digest(OUT/'source'/name)==h
    for name,h in design['trace_hashes'].items():assert object_digest(read(OUT/f'inputs/{name}.json'))==h
    hold_methods=list(dict.fromkeys(['old','decode',selection['winner']]))
    methods=dict(dev=design['methods'],holdout=hold_methods)
    runs=[read(p) for p in sorted((OUT/'runs').glob('*.json'))]
    assert len(runs)==40+8*len(hold_methods)+manifest['expected_aa']
    groups={};flat=[];checks=[]
    for r in runs:
        j=r['job'];a=audit(r);checks.append(dict(key=j['key'],**a))
        assert r['status']=='complete'
        if j['phase']=='aa':
            assert read(OUT/f"calibration_g{r['gpu']}_c{j['capacity']}.json")['passed'];continue
        assert j['phase'] in methods and j['arm'] in methods[j['phase']]
        assert j['config']['threshold']==design['arms'][j['arm']]
        assert object_digest(r['requests'])==design['trace_hashes'][j['case']]
        source=j['case'].split('_')[0]
        assert r['gpu']==manifest['bindings'][f"{j['capacity']}_{source}"]
        assert r['saved_at']>design['design_frozen_at']
        if j['phase']=='holdout':assert r['saved_at']>selection['frozen_at']
        else:
            assert digest(OUT/'runs'/f"{j['key']}.json")==selection['development_hashes'][j['key']]
            assert r['saved_at']<selection['frozen_at']
        r['_audit']=a
        groups.setdefault((j['phase'],j['case'],j['capacity']),{}).setdefault(j['arm'],[]).append(r)
        flat.append(dict(key=j['key'],phase=j['phase'],case=j['case'],capacity=j['capacity'],arm=j['arm'],
                         repeat=j['repeat'],gpu=r['gpu'],**{k:v for k,v in r['metrics'].items() if not isinstance(v,list)},**a))
    expected={(phase,c['case'],c['capacity']) for phase,field in [('dev','development'),('holdout','holdout')] for c in design[field]}
    assert set(groups)==expected
    pairs=[]
    for (phase,case,cap),arms in sorted(groups.items()):
        assert set(arms)==set(methods[phase])
        assert all(len(rs)==2 and {r['job']['repeat'] for r in rs}=={0,1} for rs in arms.values())
        def avg(arm,key):return st.fmean(r['metrics'][key] for r in arms[arm])
        for arm,rs in arms.items():
            ts=[r['metrics']['mean_ttft_ms'] for r in rs]
            row=dict(phase=phase,case=case,capacity=cap,arm=arm,
                ttft_ms=avg(arm,'mean_ttft_ms'),p95_ttft_ms=avg(arm,'p95_ttft_ms'),
                tbt_ms=avg(arm,'mean_tbt_ms'),p95_tbt_ms=avg(arm,'p95_tbt_ms'),e2e_ms=avg(arm,'mean_e2e_ms'),
                ttft_spread_pct=100*(max(ts)-min(ts))/st.fmean(ts),
                interventions=avg(arm,'interventions'),preemptions=avg(arm,'preemptions'),
                recomputed_tokens=avg(arm,'recomputed_tokens'),
                mixed_prefill_tokens=st.fmean(r['_audit']['mixed_prefill_tokens'] for r in rs),
                mixed_step_ms=st.fmean(r['_audit']['mixed_step_ms'] for r in rs))
            for reference in ('old','decode'):
                row[f'ttft_gain_{reference}_pct']=100*(1-row['ttft_ms']/avg(reference,'mean_ttft_ms'))
            pairs.append(row)
    scores={arm:-st.fmean(r['ttft_gain_old_pct'] for r in pairs if r['phase']=='dev' and r['arm']==arm)/100 for arm in methods['dev']}
    for arm in scores:assert math.isclose(scores[arm],selection['scores'][arm],abs_tol=1e-12)
    best=min(scores.values())
    assert selection['winner']==next(a for a in design['tie_order'] if scores[a]<=best+design['tie_margin'])
    aggregate={phase:{arm:dict(ttft_gain_old_pct=st.fmean(r['ttft_gain_old_pct'] for r in pairs if r['phase']==phase and r['arm']==arm),
                             ttft_gain_decode_pct=st.fmean(r['ttft_gain_decode_pct'] for r in pairs if r['phase']==phase and r['arm']==arm))
                      for arm in names} for phase,names in methods.items()}
    for name,rows in [('runs.csv',flat),('comparisons.csv',pairs),('audits.csv',checks)]:
        path=OUT/'analysis'/name;path.parent.mkdir(exist_ok=True)
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary=dict(runs=len(runs),audit=dict(passed=True,rounds=sum(a['rounds'] for a in checks)),
                 selection=selection,aggregate=aggregate,comparisons=pairs,
                 interpretation='mean TTFT only; known development windows and untouched confirmation IDs reported separately; no post-holdout retuning.')
    write(OUT/'analysis/summary.json',summary);report(summary);plot(pairs)
    print(json.dumps(dict(runs=len(runs),audit=summary['audit'],winner=selection['winner'],aggregate=aggregate),indent=2))


def report(summary):
    names=dict(old='旧式',decode='旧式+D',constant16='旧式+16',coupled_decode='D交互式',coupled_exact='有限缺口D交互式')
    lines=['# 理论公式的小规模GPU检验','',
           f"完成{summary['runs']}次GPU运行，独立审计{summary['audit']['rounds']:,}轮通过。选型仅使用mean TTFT；TBT/E2E只记录。",'',
           '开发集为已用过的13:17窗口，新确认集为15:17窗口；都是两来源、768/1536两容量、每方法两次，同一比较组同卡。新窗口与此前实验请求ID不重叠，仍来自相同日期，不能称跨天泛化。','',
           '五种结构的系数事前固定：旧式8+128x；D式再加32(1−z)；常数式再加16；D交互式(8+32(1−z))(1+16x)；有限缺口交互式(m+F)[exp((8+32(1−z))/m)−1]，m=(1+W/16)/16。均向上取整、最低1，超过可实现节约范围自动回退。','',
           f"开发集按冻结规则选中：**{names[summary['selection']['winner']]}**。先平均每单元两次TTFT，再等权平均相对旧式的百分比；距最低损失0.5个百分点内按事前简洁性顺序选择。开发分数不是独立验证成绩。",'']
    for phase,title in [('dev','开发结果'),('holdout','新窗口确认结果')]:
        lines += [f'## {title}','','正数表示mean TTFT更低；不同指标不合并打分。','',
                  '| 方法 | TTFT相对旧式改善 | TTFT相对旧+D改善 |','|---|---:|---:|']
        for arm,value in summary['aggregate'][phase].items():
            lines.append(f"| {names[arm]} | {value['ttft_gain_old_pct']:+.2f}% | {value['ttft_gain_decode_pct']:+.2f}% |")
        lines += ['','| 场景 | 容量 | 方法 | mean TTFT ms | 相对旧式改善 | 两次TTFT极差/均值 |','|---|---:|---|---:|---:|---:|']
        for row in summary['comparisons']:
            if row['phase']==phase:lines.append(f"| {row['case']} | {row['capacity']} | {names[row['arm']]} | {row['ttft_ms']:.2f} | {row['ttft_gain_old_pct']:+.2f}% | {row['ttft_spread_pct']:.2f}% |")
    lines += ['','![确认集mean TTFT](../../artifacts/v1_theory_formula_20260924_r1/figures/confirmation.png)','',
              '## 如何解释','',
              '被选中的开发赢家不自动视为最终方法，需看新窗口平均及逐场景退化。新式失败说明这组固定参数下该结构不适合本轮负载，不否定其所有校准，也不否定条件数学命题；新式成功也不证明假设的风险分布或D机制为真。',
              '本轮只有4个开发单元和4个确认单元，每单元两次。报告重复波动，不作统计显著性、全局最优或跨模型推广宣称。',
              '理论模型的共同延后成本、KV节约持续性及D影响方向尚未实测校准。所有方法沿用相同候选、排序、保护及全部特征计算成本，真实物理分配逐轮核对。',
              'CPU原生allocator回放用于可行性和动作检查，理论数值核对用于数学关系检查，都不是GPU延迟标签。',
              '[理论设计](THEORY_CN.md) · [事前方案](PROTOCOL_CN.md) · [完整CSV](../../artifacts/v1_theory_formula_20260924_r1/analysis/comparisons.csv)']
    (HERE/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')


def plot(rows):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/theory-formula-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.5))
    rs=[r for r in rows if r['phase']=='holdout'];arms=list(dict.fromkeys(r['arm'] for r in rs))
    cells=sorted({(r['case'],r['capacity']) for r in rs});width=.8/len(arms)
    for i,arm in enumerate(arms):
        selected=[next(r for r in rs if (r['case'],r['capacity'])==cell and r['arm']==arm) for cell in cells]
        ax.bar([j+(i-(len(arms)-1)/2)*width for j in range(len(cells))],
               [r['ttft_gain_old_pct'] for r in selected],width=width,label=arm)
    ax.set_xticks(range(len(cells)),[f'{c.removesuffix("_holdout")}\n{cap}' for c,cap in cells])
    ax.set_ylabel('Mean TTFT improvement over old (%)');ax.axhline(0,color='black',lw=.7)
    ax.set_title('Frozen selection: new 15:17 windows, two repeats');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/confirmation.png',dpi=180);fig.savefig(OUT/'figures/confirmation.pdf');plt.close(fig)


if __name__=='__main__':
    if '--partial' in sys.argv:
        paths=sorted((OUT/'runs').glob('*.json'))
        for p in paths:audit(read(p))
        print('Independent partial audit passed:',len(paths),'runs')
    else:main()
