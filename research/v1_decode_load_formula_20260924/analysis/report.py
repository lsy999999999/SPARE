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
from research.v1_decode_load_formula_20260924.common import *
from research.v1_budget_20260922 import run as ref
ref.OUT=OUT


def audit(r):
    ref.audit_result(r)
    assert math.isclose(st.fmean(c['finish_ms']-c['arrival_ms'] for c in r['clocks'].values()),r['metrics']['mean_e2e_ms'],abs_tol=1e-6)
    with gzip.open(OUT/r['rounds_path'],'rt') as f:rounds=json.load(f)
    normal=0;diff=0;eligible=0;blocked=0;xs=[];zs=[];ns=[]
    for row in rounds:
        state={s['rid']:s for s in row['state']}
        work=sum(max(0,s['prompt']-s['computed']) for s in state.values() if s['outputs']==0)
        held=sum(s['blocks'] for s in state.values() if s['outputs']>0)
        x=row['free_before']/(1+work/16);z=held/(row['total_blocks']-1)
        assert 0<=z<=1
        for name,value in [('work_free_ratio',x),('decoded_kv_fraction',z),('decoded_kv_headroom',1-z),('decoded_kv_blocks',held)]:
            assert math.isclose(row['features'][name],value,abs_tol=1e-12)
        spec=r['job']['config']['threshold']
        if isinstance(spec,dict):n=max(1,min(100000,math.ceil(spec['intercept']+spec['coefficients'][0]*x+spec['coefficients'][1]*(1-z))))
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
            eligible+=1;blocked+=row['scheduled']==row['baseline'];xs.append(x);zs.append(z)
            if n is not None:ns.append(n)
    return dict(rounds=len(rounds),normal_rounds=normal,different_from_old=diff,
        eligible_rounds=eligible,blocked_rounds=blocked,
        eligible_x_median=st.median(xs) if xs else None,eligible_z_median=st.median(zs) if zs else None,
        eligible_n_median=st.median(ns) if ns else None)


def main():
    assert read(OUT/'status.json')['state']=='complete'
    assert not list((OUT/'contamination').glob('*.json'))
    m=read(OUT/'manifest.json');d=read(OUT/'design.json');frozen=read(OUT/'frozen_formula.json')
    assert digest(OUT/'design.json')==m['design_sha256']
    for name,h in m['sources'].items():assert digest(OUT/'source'/name)==h
    runs=[read(p) for p in sorted((OUT/'runs').glob('*.json'))]
    assert len(runs)==88+2*len(m['gpus'])
    groups={};flat=[];checks=[]
    for r in runs:
        j=r['job'];a=audit(r);checks.append(dict(key=j['key'],**a))
        if j['phase']=='aa':continue
        assert object_digest(r['requests'])==d['trace_hashes'][j['case']]
        assert r['gpu']==m['bindings'][str(j['capacity'])]
        if j['phase']=='holdout':
            assert r['saved_at']>frozen['time']
            if j['arm']=='formula':assert j['config']['threshold']==frozen['spec']
        groups.setdefault((j['phase'],j['case'],j['capacity']),{}).setdefault(j['arm'],[]).append(r)
        flat.append(dict(key=j['key'],case=j['case'],capacity=j['capacity'],phase=j['phase'],arm=j['arm'],repeat=j['repeat'],gpu=r['gpu'],
                         **{k:v for k,v in r['metrics'].items() if not isinstance(v,list)},**a))
    comparisons=[]
    for (phase,case,cap),arms in sorted(groups.items()):
        assert set(arms)==set(d['development_methods'] if phase=='dev' else d['holdout_methods'])
        assert all(len(rs)==(2 if phase=='dev' else 1) for rs in arms.values())
        def avg(arm,metric='mean_ttft_ms'):return st.fmean(r['metrics'][metric] for r in arms[arm])
        for arm,rs in arms.items():
            ts=[r['metrics']['mean_ttft_ms'] for r in rs]
            rec=dict(phase=phase,case=case,capacity=cap,arm=arm,ttft_ms=avg(arm),
                gain_off_pct=100*(1-avg(arm)/avg('off')),gain_old_pct=100*(1-avg(arm)/avg('old')),
                repeats=len(rs),spread_pct=100*(max(ts)-min(ts))/st.fmean(ts),
                preemptions=avg(arm,'preemptions'),recomputed_tokens=avg(arm,'recomputed_tokens'),
                interventions=avg(arm,'interventions'),e2e_ms=avg(arm,'mean_e2e_ms'),tbt_ms=avg(arm,'mean_tbt_ms'),
                e2e_gain_old_pct=100*(1-avg(arm,'mean_e2e_ms')/avg('old','mean_e2e_ms')),
                tbt_gain_old_pct=100*(1-avg(arm,'mean_tbt_ms')/avg('old','mean_tbt_ms')),
                decision_ms=avg(arm,'decision_ms_mean'))
            if phase=='holdout':rec['gain_n16_pct']=100*(1-avg(arm)/avg('n16'))
            comparisons.append(rec)
    for filename,rows in [('runs.csv',flat),('comparisons.csv',comparisons),('audits.csv',checks)]:
        path=OUT/'analysis'/filename;path.parent.mkdir(exist_ok=True)
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    held=[r for r in comparisons if r['phase']=='holdout' and r['arm']=='formula']
    natural=[r for r in held if r['case'].endswith('_holdout')]
    summary=dict(frozen=frozen,runs=len(runs),audit=dict(passed=True,rounds=sum(c['rounds'] for c in checks),
        normal_rounds=sum(c['normal_rounds'] for c in checks),request_completions=sum(len(r['requests']) for r in runs)),
        holdout=held,natural_average_gain_old_pct=st.fmean(r['gain_old_pct'] for r in natural),
        natural_average_gain_off_pct=st.fmean(r['gain_off_pct'] for r in natural),
        max_natural_regression_old_pct=max(-r['gain_old_pct'] for r in natural),
        development_repeated_spreads=[r for r in comparisons if r['phase']=='dev' and r['spread_pct']>5],
        calibration_window_gpu_seconds=sum(r['elapsed_ms'] for r in runs if r['job']['phase']=='dev')/1000,
        gpu_total_window_seconds=sum(r['elapsed_ms'] for r in runs)/1000,
        provisional_adoption=frozen['provisional_adoption'],
        interpretation='Heldout single-run point estimates; signs do not prove performance guarantees or parameter transfer.')
    write(OUT/'analysis/summary.json',summary)
    plot(comparisons);report(summary,comparisons)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def plot(rows):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/decode-formula-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    for ax,source in zip(axes,('code','conversation')):
        for arm,color in [('formula','#007f73'),('old','#da8332'),('n16','#6875b8')]:
            rs=sorted((r for r in rows if r['phase']=='holdout' and r['case']==source+'_holdout' and r['arm']==arm),key=lambda r:r['capacity'])
            ax.plot([r['capacity'] for r in rs],[r['gain_off_pct'] for r in rs],label=arm,marker='o',color=color)
        ax.axhline(0,color='black',lw=.7);ax.set_title(source+' / fresh 12:17 minute');ax.set_xlabel('KV capacity (blocks)');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Mean TTFT improvement over off (%)')
    fig.suptitle('Added decode KV load: frozen coefficients, full GPU windows')
    fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/capacity_gain.png',dpi=180);fig.savefig(OUT/'figures/capacity_gain.pdf');plt.close(fig)


def report(s,rows):
    f=s['frozen'];c=f['spec']['coefficients'][1]
    lines=['# 新增decode KV占用因素：实验结果','',
        f"全部{s['runs']}次GPU运行完成，失败0；独立算术审计通过，共{s['audit']['rounds']:,}轮。18条CPU原生allocator回放通过。",'',
        f'冻结的正系数候选：`n=max(1,ceil(8+128*x+{c}*(1-z)))`，`x=F/(1+W/16)`，`z=D/(C-1)`。D为已输出首token请求当前实际持有的KV块数。',
        '旧式消融c=0保持其他实现相同；c≥0确保其他输入不变时，输出阶段持块比例越高，门槛不升。这是可解释的方向约束，不是性能定理。','',
        f"开发阶段暂可采用：{f['provisional_adoption']}。开发目标为两个代表窗口、三档容量共享系数；每个组合重复两次。选取的正c候选即使未达采用条件，也按原计划冻结后做独立验证。",'',
        '## 新窗口验证','',
        '新code/conversation各64请求，均来自12:17分钟；无上一轮输入ID重叠，但属于相同源日期。正数为改善。每个验证组合一次，结果是点估计。','',
        '| 场景 | 容量 | 新式相对off | 新式相对旧式 | 新式相对固定16 | 新式干预数 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in s['holdout']:
        lines.append(f"| {r['case']} | {r['capacity']} | {r['gain_off_pct']:+.2f}% | {r['gain_old_pct']:+.2f}% | {r['gain_n16_pct']:+.2f}% | {r['interventions']:.0f} |")
    lines += ['',f"8个自然验证单元等权平均，新式相对旧式改善 **{s['natural_average_gain_old_pct']:+.2f}%**，相对off改善 **{s['natural_average_gain_off_pct']:+.2f}%**。最坏相对旧式退化为 **{max(0,s['max_natural_regression_old_pct']):.2f}%**。这是逐单元百分比的平均，不是合并所有请求后的加速比。",'',
        '![容量曲线](../../artifacts/v1_decode_load_formula_20260924_r1/figures/capacity_gain.png)','',
        '## 重复波动与其他指标','',
        '| 开发场景 | 容量 | 方案 | 两次TTFT极差/均值 |','|---|---:|---|---:|']
    for r in s['development_repeated_spreads']:lines.append(f"| {r['case']} | {r['capacity']} | {r['arm']} | {r['spread_pct']:.2f}% |")
    if not s['development_repeated_spreads']:lines.append('| 所有开发组合 | — | — | 均不超过5% |')
    lines += ['','| 验证场景 | 容量 | E2E相对旧式改善 | TBT相对旧式改善 |','|---|---:|---:|---:|']
    for r in s['holdout']:lines.append(f"| {r['case']} | {r['capacity']} | {r['e2e_gain_old_pct']:+.2f}% | {r['tbt_gain_old_pct']:+.2f}% |")
    lines += ['','TTFT是本轮选择目标，E2E/TBT没有作为优化约束。0次干预时与off的微小墙钟差异不能归为节约动作收益。','',
        '## 校准成本与适用范围','',
        f"48次开发完整窗口累计GPU运行时长约{s['calibration_window_gpu_seconds']/60:.1f}卡分钟，不含模型加载/预热。三个并行worker的墙钟时间不能直接等于这个累计数。开发包含off控制组和所有方案两次重复；最简部署校准费用可以更少，但本轮没有以更少样本另行验证稳定性。",'',
        '只比较新增c的0、16、32三档，a=8、b=128沿用旧开发结果；所以本次是已有部署上的增量校准，不是从未知新GPU/模型开始的全参数校准证明。4096未参与本轮c选择，用同一系数验证，没有按容量重调。',
        '不能仅凭符号系数就声称已验证跨GPU、模型迁移；也不能把已输出请求持块量当作已知的未来驻留时间。独立窗口中的失败场景完整保留。','',
        '完整说明：[公式与快速校准](FORMULA_AND_CALIBRATION_CN.md)。逐单元数据：[comparisons.csv](../../artifacts/v1_decode_load_formula_20260924_r1/analysis/comparisons.csv)。审计与汇总：[summary.json](../../artifacts/v1_decode_load_formula_20260924_r1/analysis/summary.json)。']
    (HERE/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    if '--partial' in sys.argv:
        paths=sorted((OUT/'runs').glob('*.json'))
        for p in paths:audit(read(p))
        print('Independent partial audit passed:',len(paths),'completed runs')
    else:main()
