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
from research.v1_arrival_formula_20260924.common import *
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
        for name,value in [('work_free_ratio',x),('arrival_adjusted_ratio',xa),('arrival_expected_tokens',tau*rate),('arrival_tau_s',tau)]:
            assert math.isclose(row['features'][name],value,rel_tol=1e-9,abs_tol=1e-8)
        if isinstance(spec,dict):n=max(1,min(100000,math.ceil(8+128*row['features']['arrival_adjusted_ratio'])))
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
    if m.get('resource_amendment'):
        archive=OUT/m['resource_amendment']['archive']
        migration=read(archive/'migration.json')
        assert migration['new_bindings']==m['bindings']
        for record in migration['archived_files']:
            assert digest(archive/record['path'])==record['sha256']
        original=read(archive/'manifest.json')
        for key in ('sources','design_sha256','gpus','hardware','affinity'):
            assert m[key]==original[key]
    runs=[read(p) for p in sorted((OUT/'runs').glob('*.json'))]
    assert len(runs)==48+2*len(m['gpus'])
    groups={};flat=[];checks=[]
    for r in runs:
        j=r['job'];a=audit(r);checks.append(dict(key=j['key'],**a))
        if j['phase']=='aa':continue
        assert j['phase']=='compare'
        assert object_digest(r['requests'])==d['trace_hashes'][j['case']]
        source,shape=j['case'].split('_',1)
        assert r['gpu']==m['bindings'][f"{j['capacity']}_{source}"]
        assert r['saved_at']>d['design_frozen_at']
        assert j['config']['threshold']==d['arms'][j['arm']]
        r['_audit']=a
        groups.setdefault((j['case'],j['capacity']),{}).setdefault(j['arm'],[]).append(r)
        flat.append(dict(key=j['key'],source=source,shape=shape,case=j['case'],capacity=j['capacity'],arm=j['arm'],repeat=j['repeat'],gpu=r['gpu'],
            **{k:v for k,v in r['metrics'].items() if not isinstance(v,list)},**a))
    assert set(groups)=={(c['case'],c['capacity']) for c in d['cells']}
    pairs=[]
    for (case,cap),arms in sorted(groups.items()):
        assert set(arms)=={'old','arrival'}
        assert all({r['job']['repeat'] for r in rs}=={0,1} for rs in arms.values())
        def avg(arm,key):return st.fmean(r['metrics'][key] for r in arms[arm])
        source,shape=case.split('_',1)
        row=dict(case=case,source=source,shape=shape,capacity=cap)
        for metric,field in [('ttft','mean_ttft_ms'),('tbt','mean_tbt_ms'),('p95_tbt','p95_tbt_ms'),('e2e','mean_e2e_ms')]:
            for arm in ('old','arrival'):row[metric+'_'+arm+'_ms']=avg(arm,field)
            row[metric+'_gain_pct']=100*(1-avg('arrival',field)/avg('old',field))
        for arm in ('old','arrival'):
            ts=[r['metrics']['mean_ttft_ms'] for r in arms[arm]]
            row[arm+'_ttft_spread_pct']=100*(max(ts)-min(ts))/st.fmean(ts)
            row[arm+'_interventions']=avg(arm,'interventions')
            row[arm+'_preemptions']=avg(arm,'preemptions')
            row[arm+'_recomputed_tokens']=avg(arm,'recomputed_tokens')
            for key in ('different_from_old','mixed_prefill_tokens','mixed_step_ms','post_stop_5s_opportunities','post_stop_5s_different','post_stop_5s_rate_max'):
                row[arm+'_'+key]=st.fmean(r['_audit'][key] for r in arms[arm])
        pairs.append(row)
    for name,rows in [('runs.csv',flat),('comparisons.csv',pairs),('audits.csv',checks)]:
        path=OUT/'analysis'/name;path.parent.mkdir(exist_ok=True)
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    by_shape={shape:{metric:st.fmean(r[metric] for r in pairs if r['shape']==shape) for metric in
        ('ttft_gain_pct','tbt_gain_pct','p95_tbt_gain_pct','e2e_gain_pct')} for shape in ('steady','accelerating','burst_stop')}
    s=dict(runs=len(runs),audit=dict(passed=True,rounds=sum(a['rounds'] for a in checks),
        causal_event_count=sum(a['arrival_events'] for a in checks),normal_rounds=sum(a['normal_rounds'] for a in checks)),
        by_shape=by_shape,comparisons=pairs,
        interpretation='Single fixed EWMA horizon and prediction horizon; paired two-repeat controlled workload, no burst detection.')
    write(OUT/'analysis/summary.json',s);plot(pairs);report(s)
    print(json.dumps(dict(runs=len(runs),audit=s['audit'],by_shape=by_shape),ensure_ascii=False,indent=2))


def plot(rows):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/arrival-formula-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=False)
    for ax,metric in zip(axes,('ttft','tbt')):
        for i,shape in enumerate(('steady','accelerating','burst_stop')):
            rs=sorted((r for r in rows if r['shape']==shape),key=lambda r:(r['source'],r['capacity']))
            ax.bar([j+(i-1)*.25 for j in range(len(rs))],[r[metric+'_gain_pct'] for r in rs],width=.24,label=shape)
        ax.set_xticks(range(len(rs)),[r['source']+'\n'+str(r['capacity']) for r in rs]);ax.axhline(0,color='black',lw=.7)
        ax.set_title(metric.upper()+' improvement over old (%)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Arrival-rate term only: two repeats, controlled shapes, H=tau=1 s')
    fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/arrival_shapes.png',dpi=180);fig.savefig(OUT/'figures/arrival_shapes.pdf');plt.close(fig)


def report(s):
    names=dict(steady='稳定到达',accelerating='持续加速',burst_stop='单次突发后停止')
    lines=['# 旧公式单独加入A：三种到达形状的结果','',
        f"最终纳入{s['runs']}条GPU完成记录（48条正式比较、6条A/A）；独立审计通过，共{s['audit']['rounds']:,}轮、{s['audit']['causal_event_count']}个到达事件。24条CPU原生allocator回放通过。资源中断与整组重跑记录见下文。",'',
        '`n=ceil(8+128*F/(1+(W+tau*A)/16))`，新式tau=1秒，旧式tau=0；A是只由已接收prompt事件计算、H=1秒的指数平滑速率。没有D项、额外常数或双窗口突发特征；不按结果调H/tau。','',
        '## 按形状汇总','',
        '每单元同卡两次重复先取平均，再在每种形状的两个来源、两个容量之间等权平均百分比。正数改善，负数恶化。','',
        '| 形状 | TTFT改善 | 平均TBT改善 | p95 TBT改善 | E2E改善 |','|---|---:|---:|---:|---:|']
    for shape,v in s['by_shape'].items():lines.append(f"| {names[shape]} | {v['ttft_gain_pct']:+.2f}% | {v['tbt_gain_pct']:+.2f}% | {v['p95_tbt_gain_pct']:+.2f}% | {v['e2e_gain_pct']:+.2f}% |")
    interpretation=HERE/'analysis/INTERPRETATION_CN.md'
    if interpretation.exists():lines += ['',interpretation.read_text().strip(),'']
    lines += ['','![到达形状对照](../../artifacts/v1_arrival_formula_20260924_r1/figures/arrival_shapes.png)','',
        '## 逐场景结果','',
        '| 来源 | 形状 | 容量 | 旧TTFT ms | 新TTFT ms | TTFT改善 | 旧TBT ms | 新TBT ms | TBT改善 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['comparisons']:lines.append(f"| {r['source']} | {names[r['shape']]} | {r['capacity']} | {r['ttft_old_ms']:.1f} | {r['ttft_arrival_ms']:.1f} | {r['ttft_gain_pct']:+.2f}% | {r['tbt_old_ms']:.2f} | {r['tbt_arrival_ms']:.2f} | {r['tbt_gain_pct']:+.2f}% |")
    lines += ['','## 停止到达后的行为','',
        '以下只看burst_stop。在实际接收最后请求至少5秒之后，若仍存在物理可行且受保护的正节约候选，记录新式与同状态旧式的动作差异。这是新轨迹上的动作诊断，不是旧策略完整反事实TTFT。','',
        '| 来源 | 容量 | 5秒后正节约机会数 | 与同状态旧式动作不同数 | 该期间最大A tokens/s |','|---|---:|---:|---:|---:|']
    for r in s['comparisons']:
        if r['shape']=='burst_stop':lines.append(f"| {r['source']} | {r['capacity']} | {r['arrival_post_stop_5s_opportunities']:.1f} | {r['arrival_post_stop_5s_different']:.1f} | {r['arrival_post_stop_5s_rate_max']:.2f} |")
    lines += ['','## 两次重复波动','',
        '| 来源 | 形状 | 容量 | 旧TTFT极差/均值 | 新TTFT极差/均值 |','|---|---|---:|---:|---:|']
    for r in s['comparisons']:lines.append(f"| {r['source']} | {names[r['shape']]} | {r['capacity']} | {r['old_ttft_spread_pct']:.2f}% | {r['arrival_ttft_spread_pct']:.2f}% |")
    if (OUT/'recovery').exists():
        lines += ['','## 运行中断与恢复','',
            '原控制器在24次结果保存后，任务切换时发现0号卡被占用而停止。19:03按原绑定恢复，随后外部任务再次进入0号卡，保护程序终止了该worker；当时额外保存了1条结果。',
            '19:05执行资源修订：将code/768整组迁到6号卡、conversation/2048整组迁到7号卡，涉及的全部7条已保存GPU0正式结果及对应逐轮记录、污染记录均完整归档且不参与统计；不按结果优劣筛选。两个迁移组的两种方法、全部形状、两次重复在新卡完整重跑。其余组保留原卡与原结果。策略、参数、输入和方法顺序未改，归档文件哈希经独立审计。',
            '最终统计48条正式结果和6条原始A/A（共54条）；其中GPU0的2条A/A只保留为历史校验，新配对使用6、7号卡已通过的A/A。另有7条被替代的完成记录不计入结果，总共保存过61条完成记录及一次未完成的受干扰运行。当前正式结果无污染标记；不能将本实验描述为连续无中断完成。恢复和资源修订证据见artifacts的recovery目录。']
    lines += ['','## 解释范围','',
        '三个形状使用同一批请求，保持prompt/output组成和顺序；稳定/加速具有同样12秒到达跨度，单次突发则全部在t=0到达。后者同时改变形状与密度，因此不是严格固定密度的突发效应估计。均匀请求间隔也不等于异长prompt的token速率恒定。',
        '本轮使用新的14:17请求身份，但人为替换到达时间，属于受控实验。同步runner在轮次边界提交请求，A按调度器实际收到的时间计算，受到批量提交影响；不能等同于独立前端入口速率。',
        'H=1秒和tau=1秒是事前固定的机制检查值，不是已证明最优的参数。两次重复只是小规模验证，小幅差异不作显著性宣称。完整E2E、尾部、抢占、重算及干预见CSV。',
        'CPU50ms人工时钟仅用于验证到达因果性/分配，不用于报告GPU性能。审计从实际接收事件重新计算A，检查没有未接收请求进入公式，再核对W、门槛、保护及物理分配。','',
        '[事前方案](PROTOCOL_CN.md) · [比较CSV](../../artifacts/v1_arrival_formula_20260924_r1/analysis/comparisons.csv) · [审计与汇总](../../artifacts/v1_arrival_formula_20260924_r1/analysis/summary.json)']
    (HERE/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    if '--partial' in sys.argv:
        paths=sorted((OUT/'runs').glob('*.json'))
        for p in paths:audit(read(p))
        print('Independent causal/physical audit passed:',len(paths),'completed runs')
    else:main()
