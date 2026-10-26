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
from research.v1_constant_shift_20260924.common import *
from research.v1_budget_20260922 import run as ref
ref.OUT=OUT


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
        mixed_rounds=len(mixed_tokens),mixed_prefill_tokens=st.fmean(mixed_tokens) if mixed_tokens else 0,
        mixed_step_ms=st.fmean(mixed_ms) if mixed_ms else 0,
        eligible_rounds=eligible,blocked_rounds=blocked,
        eligible_x_median=st.median(xs) if xs else None,eligible_z_median=st.median(zs) if zs else None,
        eligible_n_median=st.median(ns) if ns else None)


def main():
    assert read(OUT/'status.json')['state']=='complete'
    assert not list((OUT/'contamination').glob('*.json'))
    m=read(OUT/'manifest.json');d=read(OUT/'design.json')
    assert digest(OUT/'design.json')==m['design_sha256']
    assert digest(OUT/'constant_calibration.json')==d['calibration_sha256']
    for name,h in m['sources'].items():assert digest(OUT/'source'/name)==h
    runs=[read(p) for p in sorted((OUT/'runs').glob('*.json'))]
    assert len(runs)==d['expected_formal_runs']+2*len(m['gpus'])
    groups={};flat=[];checks=[]
    for r in runs:
        j=r['job'];a=audit(r);checks.append(dict(key=j['key'],**a))
        if j['phase']=='aa':continue
        assert j['phase']=='compare'
        assert object_digest(r['requests'])==d['trace_hashes'][j['case']]
        assert r['gpu']==m['bindings'][str(j['capacity'])]
        assert r['saved_at']>d['design_frozen_at']
        assert j['config']['threshold']==d['arms'][j['arm']]
        r['_audit']=a
        groups.setdefault((j['case'],j['capacity']),{}).setdefault(j['arm'],[]).append(r)
        flat.append(dict(key=j['key'],case=j['case'],capacity=j['capacity'],arm=j['arm'],repeat=j['repeat'],gpu=r['gpu'],
            **{k:v for k,v in r['metrics'].items() if not isinstance(v,list)},**a))
    assert set(groups)=={(c['case'],c['capacity']) for c in d['fresh']+d['diagnostic']}
    pairs=[]
    for (case,cap),arms in sorted(groups.items()):
        assert set(arms)==set(d['methods'])
        assert all({r['job']['repeat'] for r in rs}=={0,1} for rs in arms.values())
        def avg(arm,key='mean_ttft_ms'):return st.fmean(r['metrics'][key] for r in arms[arm])
        for arm,rs in arms.items():
            ts=[r['metrics']['mean_ttft_ms'] for r in rs]
            row=dict(case=case,capacity=cap,arm=arm,role='fresh' if case.endswith('_fresh') else 'diagnostic',
                ttft_ms=avg(arm),tbt_ms=avg(arm,'mean_tbt_ms'),p95_tbt_ms=avg(arm,'p95_tbt_ms'),
                e2e_ms=avg(arm,'mean_e2e_ms'),p95_ttft_ms=avg(arm,'p95_ttft_ms'),
                ttft_spread_pct=100*(max(ts)-min(ts))/st.fmean(ts),
                interventions=avg(arm,'interventions'),preemptions=avg(arm,'preemptions'),
                recomputed_tokens=avg(arm,'recomputed_tokens'),decision_ms=avg(arm,'decision_ms_mean'),
                mixed_prefill_tokens=st.fmean(r['_audit']['mixed_prefill_tokens'] for r in rs),
                mixed_step_ms=st.fmean(r['_audit']['mixed_step_ms'] for r in rs))
            for refarm in ('old','dynamic'):
                for metric,field in [('ttft','mean_ttft_ms'),('tbt','mean_tbt_ms'),('p95_tbt','p95_tbt_ms'),('e2e','mean_e2e_ms')]:
                    row[f'{metric}_gain_{refarm}_pct']=100*(1-avg(arm,field)/avg(refarm,field))
            pairs.append(row)
    for name,rows in [('runs.csv',flat),('comparisons.csv',pairs),('audits.csv',checks)]:
        path=OUT/'analysis'/name;path.parent.mkdir(exist_ok=True)
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    aggregate={}
    for arm in d['methods']:
        rows=[r for r in pairs if r['role']=='fresh' and r['arm']==arm]
        aggregate[arm]={key:st.fmean(r[key] for r in rows) for key in
            ('ttft_gain_old_pct','tbt_gain_old_pct','e2e_gain_old_pct','ttft_gain_dynamic_pct','tbt_gain_dynamic_pct','e2e_gain_dynamic_pct')}
    summary=dict(runs=len(runs),formal_runs=len(flat),constant_delta=d['constant_delta'],
        audit=dict(passed=True,rounds=sum(a['rounds'] for a in checks),normal_rounds=sum(a['normal_rounds'] for a in checks)),
        fresh_equal_cell_average=aggregate,comparisons=pairs,
        repeats_above_5pct=[r for r in pairs if r['ttft_spread_pct']>5],
        interpretation='Constant matched on old development feature magnitudes, not TTFT-optimized. Two repetitions; diagnostic reuse separated.')
    write(OUT/'analysis/summary.json',summary)
    plot(pairs);report(summary)
    print(json.dumps(dict(runs=len(runs),audit=summary['audit'],fresh_average=aggregate),ensure_ascii=False,indent=2))


def plot(rows):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/constant-formula-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    colors=dict(dynamic='#007f73',constant_matched='#dd8833',constant32='#6875b8')
    for col,source in enumerate(('code','conversation')):
        for row,metric in enumerate(('ttft','tbt')):
            ax=axes[row,col]
            for arm,color in colors.items():
                rs=sorted((r for r in rows if r['case']==source+'_fresh' and r['arm']==arm),key=lambda r:r['capacity'])
                ax.plot([r['capacity'] for r in rs],[r[metric+'_gain_old_pct'] for r in rs],marker='o',label=arm,color=color)
            ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.2);ax.set_title(source+' / '+metric.upper());ax.legend(fontsize=8)
            ax.set_ylabel('Improvement over old (%)')
            if row==1:ax.set_xlabel('KV capacity (blocks)')
    fig.suptitle('Constant offset vs decode-state correction: two repeats, fresh windows')
    fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/ttft_tbt.png',dpi=180);fig.savefig(OUT/'figures/ttft_tbt.pdf');plt.close(fig)


def report(s):
    lines=['# 固定加常数对照实验结果','',f"{s['runs']}次GPU运行全部完成，失败0，独立审计通过，共{s['audit']['rounds']:,}轮。CPU16条完整原生allocator回放通过。",'',
        '四种方法：old=`8+128x`；dynamic=`8+128x+32(1-z)`；constant_matched=`8+128x+16`；constant32=`8+128x+32`。均向上取整并保持原有保护与候选顺序。',
        '常数16由11:17开发日志上动态加项的等权均值15.51149四舍五入得到，没有用本轮GPU结果挑选。它匹配旧状态上的平均加项，不能保证新轨迹上的相同干预强度。','',
        '## 新窗口结果','',
        '13:17两个新分钟窗口，各64请求，768/1536/2048；每个方法同卡配对两次。以下为先平均每单元两次指标，再在6个新单元间平均相对百分比。正数表示更快，负数更慢。','',
        '| 方法 | TTFT相对old改善 | TBT相对old改善 | E2E相对old改善 | TTFT相对dynamic改善 | TBT相对dynamic改善 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,v in s['fresh_equal_cell_average'].items():
        lines.append(f"| {arm} | {v['ttft_gain_old_pct']:+.2f}% | {v['tbt_gain_old_pct']:+.2f}% | {v['e2e_gain_old_pct']:+.2f}% | {v['ttft_gain_dynamic_pct']:+.2f}% | {v['tbt_gain_dynamic_pct']:+.2f}% |")
    lines += ['','![TTFT与TBT对照](../../artifacts/v1_constant_shift_20260924_r1/figures/ttft_tbt.png)','',
        '## 逐场景数据','',
        '| 场景 | 容量 | 方法 | TTFT ms | TBT ms | p95 TBT ms | E2E ms | 干预数 |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in s['comparisons']:
        lines.append(f"| {r['case']} | {r['capacity']} | {r['arm']} | {r['ttft_ms']:.1f} | {r['tbt_ms']:.2f} | {r['p95_tbt_ms']:.2f} | {r['e2e_ms']:.1f} | {r['interventions']:.1f} |")
    lines += ['','code_diagnostic是已知12:17 code/1536场景的复现，不计入上面的新窗口平均。每单元只有两次重复，小幅差异不能据此作显著性断言。','',
        '## 超过5%的TTFT重复波动','',
        '| 场景 | 容量 | 方法 | 极差/均值 |','|---|---:|---|---:|']
    for r in s['repeats_above_5pct']:lines.append(f"| {r['case']} | {r['capacity']} | {r['arm']} | {r['ttft_spread_pct']:.2f}% |")
    if not s['repeats_above_5pct']:lines.append('| 全部单元 | — | — | 均≤5% |')
    lines += ['','## 解释边界','',
        '本轮没有使用新结果拟合最优常数；常数16的相近表现会削弱“必须输入D”的证据，但不能据此证明D永远无用。两个常数失败也不能证明任何常数都无法替代动态式。所有判断都应同时看TTFT、TBT、E2E的取舍，不能只根据TTFT宣布全面更好。',
        '四种方法计算相同特征/诊断，候选与保护相同；保护decode token推进不保证混合轮次耗时不增加。混合轮次的prefill规模/时长、抢占与重算见CSV，可用于解释观察到的变化，不单独当作因果证明。','',
        '[事前设计](PROTOCOL_CN.md) · [完整CSV](../../artifacts/v1_constant_shift_20260924_r1/analysis/comparisons.csv) · [审计汇总](../../artifacts/v1_constant_shift_20260924_r1/analysis/summary.json)']
    (HERE/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    if '--partial' in sys.argv:
        paths=sorted((OUT/'runs').glob('*.json'))
        for p in paths:audit(read(p))
        print('Independent partial audit passed:',len(paths),'completed runs')
    else:main()
