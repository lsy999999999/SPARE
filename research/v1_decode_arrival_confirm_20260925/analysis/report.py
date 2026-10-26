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
from research.v1_decode_arrival_confirm_20260925.common import *
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
        if isinstance(spec,dict):n=max(1,min(100000,math.ceil(8+128*xa+spec['coefficients'][1]*(1-z))))
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
    assert {(r['gpu'],r['job']['capacity']) for r in runs if r['job']['phase']=='aa'}==aa_pairs
    groups={};flat=[];checks=[]
    for r in runs:
        j=r['job'];a=audit(r);checks.append(dict(key=j['key'],**a));assert r['status']=='complete'
        if j['phase']=='aa':
            assert j['key'] in {f'aa_g{g}_c{cap}_r{rep}' for g,cap in aa_pairs for rep in (0,1)}
            assert read(OUT/f"calibration_g{r['gpu']}_c{j['capacity']}.json")['passed'];continue
        assert j['phase']=='compare' and j['arm'] in d['methods']
        assert j['config']['threshold']==d['arms'][j['arm']]
        assert object_digest(r['requests'])==d['trace_hashes'][j['case']]
        source,shape=j['case'].split('_',1)
        assert r['gpu']==m['bindings'][f"{j['capacity']}_{source}"]
        assert r['saved_at']>d['design_frozen_at']
        r['_audit']=a
        groups.setdefault((j['case'],j['capacity']),{}).setdefault(j['arm'],[]).append(r)
        flat.append(dict(key=j['key'],case=j['case'],source=source,shape=shape,capacity=j['capacity'],arm=j['arm'],
                         repeat=j['repeat'],gpu=r['gpu'],**{k:v for k,v in r['metrics'].items() if not isinstance(v,list)},**a))
    assert set(groups)=={(c['case'],c['capacity']) for c in d['cells']}
    pairs=[];interactions=[]
    for (case,cap),arms in sorted(groups.items()):
        assert set(arms)==set(d['methods'])
        assert all(len(rs)==2 and {r['job']['repeat'] for r in rs}=={0,1} for rs in arms.values())
        def avg(arm,key='mean_ttft_ms'):return st.fmean(r['metrics'][key] for r in arms[arm])
        source,shape=case.split('_',1)
        for arm,rs in arms.items():
            ts=[r['metrics']['mean_ttft_ms'] for r in rs]
            row=dict(case=case,source=source,shape=shape,capacity=cap,arm=arm,ttft_ms=avg(arm),
                ttft_spread_pct=100*(max(ts)-min(ts))/st.fmean(ts),tbt_ms=avg(arm,'mean_tbt_ms'),
                p95_tbt_ms=avg(arm,'p95_tbt_ms'),e2e_ms=avg(arm,'mean_e2e_ms'),p95_ttft_ms=avg(arm,'p95_ttft_ms'),
                interventions=avg(arm,'interventions'),preemptions=avg(arm,'preemptions'),recomputed_tokens=avg(arm,'recomputed_tokens'))
            for refarm in ('old','decode','arrival'):
                row[f'ttft_gain_{refarm}_pct']=100*(1-avg(arm)/avg(refarm))
            for key in ('mixed_prefill_tokens','mixed_step_ms','different_from_old','post_stop_5s_opportunities','post_stop_5s_different'):
                row[key]=st.fmean(r['_audit'][key] for r in rs)
            pairs.append(row)
        effects=[]
        for repeat in (0,1):
            ts={arm:next(r['metrics']['mean_ttft_ms'] for r in rs if r['job']['repeat']==repeat) for arm,rs in arms.items()}
            effects.append(ts['combined']-ts['decode']-ts['arrival']+ts['old'])
        interaction=avg('combined')-avg('decode')-avg('arrival')+avg('old')
        interactions.append(dict(case=case,shape=shape,source=source,capacity=cap,
            interaction_ms=interaction,interaction_pct_of_old=100*interaction/avg('old'),
            repeat0_ms=effects[0],repeat1_ms=effects[1],
            combined_gain_vs_best_single_pct=100*(1-avg('combined')/min(avg(a) for a in ('old','decode','arrival')))))
    scopes=('all','steady','accelerating','burst_stop','capacity_768','capacity_1536')
    def in_scope(r,scope):
        return scope=='all' or r['shape']==scope or scope==f"capacity_{r['capacity']}"
    aggregates={scope:{arm:{key:st.fmean(r[key] for r in pairs if r['arm']==arm and in_scope(r,scope))
                                for key in ('ttft_gain_old_pct','ttft_gain_decode_pct','ttft_gain_arrival_pct')}
                       for arm in d['methods']} for scope in scopes}
    for name,rows in [('runs.csv',flat),('comparisons.csv',pairs),('audits.csv',checks),('interactions.csv',interactions)]:
        p=OUT/'analysis'/name;p.parent.mkdir(exist_ok=True)
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with p.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary=dict(runs=len(runs),audit=dict(passed=True,rounds=sum(a['rounds'] for a in checks),
                arrival_events=sum(a['arrival_events'] for a in checks)),aggregates=aggregates,
                comparisons=pairs,interactions=interactions,
                scope='Fixed 768/1536 blocks, two fresh 17:17 sources, controlled arrivals, two repeats; mean TTFT only.')
    write(OUT/'analysis/summary.json',summary);report(summary);plot(pairs)
    print(json.dumps(dict(runs=len(runs),audit=summary['audit'],aggregates=aggregates),indent=2))


def report(summary):
    names=dict(old='旧式',decode='旧式+D',arrival='旧式+A',combined='旧式+D+A')
    shapes=dict(all='全部十二单元',steady='稳定到达',accelerating='持续加速',burst_stop='突发后停止',capacity_768='768块',capacity_1536='1536块')
    lines=['# D+A公式独立确认：新窗口与768/1536容量','',
           f"完成{summary['runs']}次GPU运行，独立审计{summary['audit']['rounds']:,}轮、{summary['audit']['arrival_events']}个到达事件通过。",'',
           '组合式为n=ceil(8+128F/(1+(W+tau*A)/16)+k(1-D/(C-1)))。旧式tau=k=0；D式tau=0,k=32；A式tau=1秒,k=0；组合tau=1秒,k=32。所有方法最低1，沿用物理候选和保护。A为prompt tokens/秒的H=1秒指数平滑速率。系数事前固定，不用本轮结果调参。','',
           '**唯一判断指标为mean TTFT。** TBT/E2E记录在CSV，不抵偿TTFT退化。768/1536两种容量、两个来源、三种受控到达形状、每方法两次。先平均每单元两次TTFT，再等权平均相对变化；不是合并请求后的加速比。','']
    for scope,title in shapes.items():
        lines += [f'## {title}','','正数表示mean TTFT更低。','',
                  '| 方法 | 相对旧式改善 | 相对旧+D改善 | 相对旧+A改善 |','|---|---:|---:|---:|']
        for arm,v in summary['aggregates'][scope].items():
            lines.append(f"| {names[arm]} | {v['ttft_gain_old_pct']:+.2f}% | {v['ttft_gain_decode_pct']:+.2f}% | {v['ttft_gain_arrival_pct']:+.2f}% |")
    lines += ['','## 逐场景结果与重复波动','','| 容量 | 来源 | 形状 | 方法 | mean TTFT ms | 相对旧式改善 | 两次极差/均值 |','|---:|---|---|---|---:|---:|---:|']
    for r in summary['comparisons']:
        lines.append(f"| {r['capacity']} | {r['source']} | {shapes[r['shape']]} | {names[r['arm']]} | {r['ttft_ms']:.2f} | {r['ttft_gain_old_pct']:+.2f}% | {r['ttft_spread_pct']:.2f}% |")
    lines += ['','## 两项是否有交互','','I=T(D+A)-T(D)-T(A)+T(old)。负值表示在此TTFT加性尺度上，加入D后A的效果更有利；不能仅凭负值证明信息独立或机制普遍成立。正交实验的两次结果均列出，不作显著性断言。','',
              '| 容量 | 来源 | 形状 | I ms | I/旧TTFT | 第1次I ms | 第2次I ms | 组合相对事后最佳单项改善 |','|---:|---|---|---:|---:|---:|---:|---:|']
    for r in summary['interactions']:
        lines.append(f"| {r['capacity']} | {r['source']} | {shapes[r['shape']]} | {r['interaction_ms']:+.2f} | {r['interaction_pct_of_old']:+.2f}% | {r['repeat0_ms']:+.2f} | {r['repeat1_ms']:+.2f} | {r['combined_gain_vs_best_single_pct']:+.2f}% |")
    lines += ['','事后最佳单项在old/D/A三者中逐场景选择，仅作经验参照，不是一个可在线部署的策略或理论最优。','',
              '![组合相对单项](../../artifacts/v1_decode_arrival_confirm_20260925_r1/figures/combined_ttft.png)','',
              '## 结论边界','',
              '这是固定参数的一次组合检验，没有寻找全局最优。组合优于旧式但不优于D或A，不能证明需要同时输入D和A；需要看相对两个单项及逐场景退化。',
              '请求取自新的17:17身份，三种形状复用相同请求组成。稳定与加速到达跨度均12秒；突发后停止全部在t=0到达，因此同时改变形状与密度。均匀请求间隔不等于恒定token速率。',
              'A只由已经提交到scheduler的请求构造，同步runner按调度轮边界提交，不能等同独立前端入口速率。没有未来到达或真实剩余输出泄漏，没有新突发窗口特征。',
              '本轮覆盖768/1536块；2048结果来自上一轮不同请求窗口，不能把两轮差异解释为容量的单因素效应，也不能外推其他模型/卡或自然到达trace。相同来源日期的新分钟不是跨天验证。两次重复支持初步方向检查，不构成统计显著性证据。',
              'CPU人工50ms时钟只验证因果性、原生分配与消融等价，不是GPU性能仿真。原始TTFT、TBT、E2E、抢占/重算、门槛和动作诊断均保留。','',
              '[事前方案](PROTOCOL_CN.md) · [比较CSV](../../artifacts/v1_decode_arrival_confirm_20260925_r1/analysis/comparisons.csv) · [交互CSV](../../artifacts/v1_decode_arrival_confirm_20260925_r1/analysis/interactions.csv)']
    (HERE/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')


def plot(rows):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/decode-arrival-confirm-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True)
    for ai,cap in enumerate((768,1536)):
        for aj,source in enumerate(('code','conversation')):
            ax=axes[ai,aj]
            for i,arm in enumerate(('decode','arrival','combined')):
                selected=[next(r for r in rows if r['capacity']==cap and r['source']==source and r['shape']==shape and r['arm']==arm)
                          for shape in ('steady','accelerating','burst_stop')]
                ax.bar([j+(i-1)*.25 for j in range(3)],[r['ttft_gain_old_pct'] for r in selected],width=.24,label=arm)
            ax.set_xticks(range(3),['steady','accelerating','burst stop']);ax.axhline(0,color='black',lw=.7)
            ax.set_title(f'{source} / {cap} blocks');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
        axes[ai,0].set_ylabel('Mean TTFT improvement over old (%)')
    fig.suptitle('Frozen D+A confirmation: fresh requests, two repetitions');fig.tight_layout()
    (OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/combined_ttft.png',dpi=180);fig.savefig(OUT/'figures/combined_ttft.pdf');plt.close(fig)


if __name__=='__main__':
    if '--partial' in sys.argv:
        paths=sorted((OUT/'runs').glob('*.json'))
        for p in paths:audit(read(p))
        print('Independent causal/physical audit passed:',len(paths),'runs')
    else:main()
