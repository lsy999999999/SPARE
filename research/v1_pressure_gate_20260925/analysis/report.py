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
from research.v1_pressure_gate_20260925.common import *
from research.v1_budget_20260922 import run as ref
ref.OUT=OUT


def audit(r):
    ref.audit_result(r)
    assert math.isclose(st.fmean(c['finish_ms']-c['arrival_ms'] for c in r['clocks'].values()),r['metrics']['mean_e2e_ms'],abs_tol=1e-6)
    with gzip.open(OUT/r['rounds_path'],'rt') as f:rounds=json.load(f)
    normal=0;diff=0;eligible=0;blocked=0;xs=[];rates=[];ns=[];mixed_tokens=[];mixed_ms=[]
    events=[];seen=set();reqs={q['id']:q for q in r['requests']};previous_time=0.;post_stop=0;post_stop_diff=0;tail_rates=[]
    for row in rounds:
        state={s['rid']:s for s in row['state']}
        prefill=sum(n for rid,n in row['scheduled'].items() if state[rid]['outputs']==0)
        decode=sum(1 for rid in row['scheduled'] if state[rid]['outputs']>0 and state[rid]['tokens']-state[rid]['computed']==1)
        if prefill and decode:mixed_tokens.append(prefill);mixed_ms.append(row['step_ms'])
        work=sum(max(0,s['prompt']-s['computed']) for s in state.values() if s['outputs']==0)
        t=row['features']['arrival_time_s']
        assert t>=previous_time
        for e in row['arrival_events']:
            assert e['id'] not in seen and e['id'] in reqs
            assert e['prompt']==reqs[e['id']]['prompt']
            assert previous_time<=e['time_s']<=t
            seen.add(e['id']);events.append(e)
        previous_time=t
        assert set(state)<=seen
        rate=sum(e['prompt']*math.exp(-(t-e['time_s'])) for e in events)
        assert math.isclose(row['features']['arrival_rate'],rate,rel_tol=1e-9,abs_tol=1e-8)
        assert row['features']['arrival_observed_tokens']==sum(e['prompt'] for e in events)
        spec=r['job']['config']['threshold']
        tau=spec.get('tau_s',0.) if isinstance(spec,dict) else 0.
        x=row['free_before']/(1+work/16)
        xa=row['free_before']/(1+(work+tau*rate)/16)
        held=sum(s['blocks'] for s in state.values() if s['outputs']>0)
        z=held/(row['total_blocks']-1);assert 0<=z<=1
        for name,value in [('work_free_ratio',x),('arrival_adjusted_ratio',xa),('arrival_expected_tokens',tau*rate),('arrival_tau_s',tau),
                           ('decoded_kv_blocks',held),('decoded_kv_fraction',z),('decoded_kv_headroom',1-z)]:
            assert math.isclose(row['features'][name],value,rel_tol=1e-9,abs_tol=1e-8)
        gate=xa/(1+xa)
        assert math.isclose(row['features']['pressure_gate'],gate,abs_tol=1e-10)
        assert math.isclose(row['features']['pressure_decoded_headroom'],(1-z)*gate,abs_tol=1e-10)
        if isinstance(spec,dict):
            ext={'decoded_kv_headroom':1-z,'pressure_decoded_headroom':(1-z)*gate,'pressure_gate':gate}[spec['features'][1]]
            n=max(1,min(100000,math.ceil(8+128*xa+spec['coefficients'][1]*ext)))
        else:n=None if spec=='off' else spec
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
            eligible+=1;blocked+=row['scheduled']==row['baseline'];xs.append(x);rates.append(rate)
            if len(seen)==len(reqs) and t-events[-1]['time_s']>=5:
                post_stop+=1;post_stop_diff+=action(old)!=row['scheduled'];tail_rates.append(rate)
            if n is not None:ns.append(n)
    assert seen==set(reqs)
    return dict(rounds=len(rounds),normal_rounds=normal,different_from_old=diff,
        post_stop_5s_opportunities=post_stop,post_stop_5s_different=post_stop_diff,
        post_stop_5s_rate_max=max(tail_rates,default=0),arrival_events=len(events),
        mixed_rounds=len(mixed_tokens),mixed_prefill_tokens=st.fmean(mixed_tokens) if mixed_tokens else 0,
        mixed_step_ms=st.fmean(mixed_ms) if mixed_ms else 0,
        eligible_rounds=eligible,blocked_rounds=blocked,
        eligible_x_median=st.median(xs) if xs else None,eligible_rate_median=st.median(rates) if rates else None,
        eligible_n_median=st.median(ns) if ns else None)


def main():
    assert read(OUT/'status.json')['state']=='complete'
    assert not list((OUT/'contamination').glob('*.json'))
    m=read(OUT/'manifest.json');d=read(OUT/'design.json')
    assert digest(OUT/'design.json')==m['design_sha256']
    for name,h in m['sources'].items():assert digest(OUT/'source'/name)==h
    for name,h in d['trace_hashes'].items():assert object_digest(read(OUT/f'inputs/{name}.json'))==h
    runs=[read(p) for p in sorted((OUT/'runs').glob('*.json'))]
    aa_pairs={(g,int(key.split('_')[0])) for key,g in m['bindings'].items()}
    assert len(runs)==d['expected_formal_runs']+2*len(aa_pairs)
    assert {r['job']['key'] for r in runs if r['job']['phase']=='aa'}=={f'aa_g{g}_c{cap}_r{rep}' for g,cap in aa_pairs for rep in (0,1)}
    groups={};flat=[];checks=[]
    for r in runs:
        j=r['job'];assert r['status']=='complete';a=audit(r);checks.append(dict(key=j['key'],**a))
        if j['phase']=='aa':
            assert read(OUT/f"calibration_g{r['gpu']}_c{j['capacity']}.json")['passed'];continue
        assert j['phase']=='compare' and j['arm'] in d['methods']
        assert j['config']['threshold']==d['arms'][j['arm']]
        assert object_digest(r['requests'])==d['trace_hashes'][j['case']]
        assert r['gpu']==m['bindings'][f"{j['capacity']}_{j['source']}"]
        assert r['saved_at']>d['design_frozen_at']
        cell=next(c for c in d['cells'] if c['case']==j['case'] and c['capacity']==j['capacity'])
        assert all(j[k]==v for k,v in cell.items())
        r['_audit']=a
        groups.setdefault((j['case'],j['capacity']),{}).setdefault(j['arm'],[]).append(r)
        flat.append(dict(key=j['key'],**cell,arm=j['arm'],repeat=j['repeat'],gpu=r['gpu'],
                         **{k:v for k,v in r['metrics'].items() if not isinstance(v,list)},**a))
    assert set(groups)=={(c['case'],c['capacity']) for c in d['cells']}
    pairs=[]
    for (case,cap),arms in sorted(groups.items()):
        assert set(arms)==set(d['methods'])
        assert all(len(rs)==2 and {r['job']['repeat'] for r in rs}=={0,1} for rs in arms.values())
        def avg(arm,key='mean_ttft_ms'):return st.fmean(r['metrics'][key] for r in arms[arm])
        cell=next(c for c in d['cells'] if c['case']==case and c['capacity']==cap)
        for arm,rs in arms.items():
            ts=[r['metrics']['mean_ttft_ms'] for r in rs]
            row=dict(**cell,arm=arm,ttft_ms=avg(arm),ttft_spread_pct=100*(max(ts)-min(ts))/st.fmean(ts))
            for key in ('mean_tbt_ms','p95_tbt_ms','mean_e2e_ms','p95_ttft_ms','interventions','preemptions','recomputed_tokens'):row[key]=avg(arm,key)
            for refarm in d['methods']:
                row[f'gain_{refarm}_pct']=100*(1-avg(arm)/avg(refarm))
                for rep in (0,1):
                    num=next(r['metrics']['mean_ttft_ms'] for r in rs if r['job']['repeat']==rep)
                    den=next(r['metrics']['mean_ttft_ms'] for r in arms[refarm] if r['job']['repeat']==rep)
                    row[f'gain_{refarm}_r{rep}_pct']=100*(1-num/den)
            for key in ('mixed_prefill_tokens','mixed_step_ms','different_from_old'):row[key]=st.fmean(r['_audit'][key] for r in rs)
            pairs.append(row)
    aggregates={}
    for window in ('reused','fresh'):
        for cap in (None,768,1536):
            label=f'{window}_{cap or "all"}'
            aggregates[label]={arm:{f'gain_{refarm}_pct':st.fmean(r[f'gain_{refarm}_pct'] for r in pairs if
                r['window']==window and r['arm']==arm and (cap is None or r['capacity']==cap))
                for refarm in d['methods']} for arm in d['methods']}
    for name,rows in [('runs.csv',flat),('comparisons.csv',pairs),('audits.csv',checks)]:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (OUT/'analysis'/name).open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary=dict(runs=len(runs),audit=dict(passed=True,rounds=sum(a['rounds'] for a in checks),arrival_events=sum(a['arrival_events'] for a in checks)),
                 comparisons=pairs,aggregates=aggregates,scope=d['scope'])
    write(OUT/'analysis/summary.json',summary);report(summary);plot(pairs)
    print(json.dumps(dict(runs=len(runs),audit=summary['audit'],aggregates=aggregates),indent=2))


def report(s):
    names=dict(old='旧式',arrival='旧+A',combined='旧+D+A',pressure_decode='压力衰减D+A',pressure_constant='压力衰减常数+A')
    lines=['# 资源压力与D修正：机制对照结果','',
           f"完成{s['runs']}次运行，独立审计{s['audit']['rounds']:,}轮、{s['audit']['arrival_events']}个到达事件通过。",'',
           '令x=F/(1+(W+A)/16)、u=1-D/(C-1)、g=x/(1+x)。新候选为ceil(8+128x+32ug)，无D对照为ceil(8+128x+32g)，原组合为ceil(8+128x+32u)。均最低1；A与全部参数沿用冻结值。','',
           '唯一主指标mean TTFT。正值表示改善。先平均每单元两次，再等权平均相对百分比；复用17:17诊断和新18:17确认分别报告，不混合选择赢家。两种来源各取一种形状，不能外推全部到达形状。','']
    for label,arms in s['aggregates'].items():
        lines += [f'## {label}','','| 方法 | 相对旧式 | 相对A | 相对原组合 | 相对压力常数 |','|---|---:|---:|---:|---:|']
        for arm,v in arms.items():lines.append(f"| {names[arm]} | {v['gain_old_pct']:+.2f}% | {v['gain_arrival_pct']:+.2f}% | {v['gain_combined_pct']:+.2f}% | {v['gain_pressure_constant_pct']:+.2f}% |")
    lines += ['','## 逐场景与波动','','| 窗口 | 来源/形状 | 容量 | 方法 | TTFT ms | 相对旧式 | 两次极差/均值 | 抢占 | 重算tokens |','|---|---|---:|---|---:|---:|---:|---:|---:|']
    for r in s['comparisons']:
        lines.append(f"| {r['window']} | {r['source']}/{r['shape']} | {r['capacity']} | {names[r['arm']]} | {r['ttft_ms']:.2f} | {r['gain_old_pct']:+.2f}% | {r['ttft_spread_pct']:.2f}% | {r['preemptions']:.1f} | {r['recomputed_tokens']:.1f} |")
    lines += ['','## 新函数相对三个关键对照：两次重复','','| 窗口 | 来源/容量 | vs A 两次 | vs 原组合 两次 | vs 压力常数 两次 |','|---|---|---:|---:|---:|']
    for r in s['comparisons']:
        if r['arm']!='pressure_decode':continue
        vals=[' / '.join(f"{r[f'gain_{arm}_r{rep}_pct']:+.2f}%" for rep in (0,1)) for arm in ('arrival','combined','pressure_constant')]
        lines.append(f"| {r['window']} | {r['source']}/{r['capacity']} | {' | '.join(vals)} |")
    lines += ['','![压力修正对照](../../artifacts/v1_pressure_gate_20260925_r1/figures/pressure_ttft.png)','',
        '复用场景依据既往结果选择，因此属于机制诊断；fresh场景系数和形式在新GPU结果前冻结，但仍只有两个来源窗口，不能作广泛独立样本或统计显著性断言。无burst-stop测试。',
        'gate改变整个闭环轨迹；抢占/重算变化只为机制线索，不是完整因果分解。若改善原组合却不超过A，不能证明仍需要D；若无D压力常数同样好，不能归功于D的独立信息。',
        '同组固定GPU，不同容量可能不同卡。TBT/E2E只记录，不用于筛选或抵偿TTFT退化。CPU人工时钟仅用于正确性。','',
        '[事前方案](PROTOCOL_CN.md) · [比较CSV](../../artifacts/v1_pressure_gate_20260925_r1/analysis/comparisons.csv)']
    (HERE/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')


def plot(rows):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/pressure-gate-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for ai,window in enumerate(('reused','fresh')):
        for aj,cap in enumerate((768,1536)):
            ax=axes[ai,aj]
            for i,arm in enumerate(('arrival','combined','pressure_decode','pressure_constant')):
                vals=[next(r['gain_old_pct'] for r in rows if r['window']==window and r['capacity']==cap and r['source']==source and r['arm']==arm) for source in ('code','conversation')]
                ax.bar([j+(i-1.5)*.2 for j in range(2)],vals,width=.19,label=arm)
            ax.set_xticks([0,1],['code accelerating','conversation steady']);ax.axhline(0,color='black',lw=.7)
            ax.set_title(f'{window} / {cap} blocks');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
        axes[ai,0].set_ylabel('Mean TTFT improvement over old (%)')
    fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(OUT/f'figures/pressure_ttft.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    if '--partial' in sys.argv:
        paths=sorted((OUT/'runs').glob('*.json'))
        for p in paths:audit(read(p))
        print('Independent audit passed:',len(paths),'runs')
    else:main()
