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
from research.v1_broad12h_20260925.common import *
from research.v1_budget_20260922 import run as ref
ref.OUT=OUT


def audit(r):
    ref.audit_result(r)
    assert math.isclose(st.fmean(c['finish_ms']-c['arrival_ms'] for c in r['clocks'].values()),r['metrics']['mean_e2e_ms'],abs_tol=1e-6)
    with gzip.open(OUT/r['rounds_path'],'rt') as f:rounds=json.load(f)
    coverage=dict(rounds=0,free_lt_10pct=0,free_gt_90pct=0,prefill_dominant=0,decode_dominant=0,recomputing=0)
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
        preheld=sum(s['blocks'] for s in state.values() if s['outputs']==0)
        cu=row['total_blocks']-1
        assert held+preheld+row['free_before']==cu
        coverage['rounds']+=1
        coverage['free_lt_10pct']+=row['free_before']/cu<.1
        coverage['free_gt_90pct']+=row['free_before']/cu>.9
        coverage['prefill_dominant']+=preheld/cu>.5
        coverage['decode_dominant']+=held/cu>.5
        coverage['recomputing']+=row['features']['recomputing']>0
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
    return dict(state_coverage=coverage,rounds=len(rounds),normal_rounds=normal,different_from_old=diff,
        post_stop_5s_opportunities=post_stop,post_stop_5s_different=post_stop_diff,
        post_stop_5s_rate_max=max(tail_rates,default=0),arrival_events=len(events),
        mixed_rounds=len(mixed_tokens),mixed_prefill_tokens=st.fmean(mixed_tokens) if mixed_tokens else 0,
        mixed_step_ms=st.fmean(mixed_ms) if mixed_ms else 0,
        eligible_rounds=eligible,blocked_rounds=blocked,
        eligible_x_median=st.median(xs) if xs else None,eligible_rate_median=st.median(rates) if rates else None,
        eligible_n_median=st.median(ns) if ns else None)


def main(partial=False):
    m=read(OUT/'manifest.json');d=read(OUT/'design.json');status=read(OUT/'status.json')
    assert digest(OUT/'design.json')==m['design_sha256']
    for name,h in m['sources'].items():assert digest(OUT/'source'/name)==h
    for name,h in d['trace_hashes'].items():assert object_digest(read(OUT/f'inputs/{name}.json'))==h
    cells={f"{c['case']}_c{c['capacity']}":c for c in d['cells']}
    accepted=[];failures=[];pairs=[];flat=[];checks=[];calibrated=set();coverage={}
    group_paths=sorted((OUT/'groups').glob('*.json'))
    for p in group_paths:
        entry=read(p)
        assert entry['group'] in cells and entry['cell']==cells[entry['group']]
        if entry['state']!='complete':failures.append(entry);continue
        folder=OUT/entry['folder'];assert not (folder/'contamination.json').exists()
        assert read(folder/'attempt_status.json')==entry and read(folder/'worker_complete.json')['passed']
        task=read(folder/'task.json');cell=entry['cell'];runs=[read(p) for p in sorted((folder/'runs').glob('*.json'))]
        assert len(runs)==len(d['methods'])*d['repeats']
        expected={j['key']:j for j in task['jobs']};assert {r['job']['key'] for r in runs}==set(expected)
        calkey=entry['gpu'],cell['capacity']
        cp=OUT/f'calibration/g{calkey[0]}_c{calkey[1]}'
        assert read(cp/'status.json')['passed']
        if calkey not in calibrated:
            cr=[read(p) for p in sorted((cp/'runs').glob('*.json'))];assert len(cr)==2
            for r in cr:checks.append(dict(key=r['job']['key'],**audit(r)))
            calibrated.add(calkey)
        arms={}
        for r in runs:
            j=r['job'];assert r['status']=='complete' and j==expected[j['key']]
            assert r['gpu']==entry['gpu'] and object_digest(r['requests'])==d['trace_hashes'][cell['case']]
            assert j['config']['threshold']==d['arms'][j['arm']]
            assert r['saved_at']>=d['design_frozen_at']
            a=audit(r);checks.append(dict(key=j['key'],**a));arms.setdefault(j['arm'],[]).append(r)
            for key,value in a['state_coverage'].items():coverage[key]=coverage.get(key,0)+value
            flat.append(dict(group=entry['group'],**cell,arm=j['arm'],repeat=j['repeat'],gpu=r['gpu'],
                **{k:v for k,v in r['metrics'].items() if not isinstance(v,list)},**{k:v for k,v in a.items() if not isinstance(v,dict)}))
        assert set(arms)==set(d['methods'])
        assert all(len(rs)==2 and {r['job']['repeat'] for r in rs}=={0,1} for rs in arms.values())
        def avg(arm,key='mean_ttft_ms'):return st.fmean(r['metrics'][key] for r in arms[arm])
        for arm,rs in arms.items():
            ts=[r['metrics']['mean_ttft_ms'] for r in rs]
            row=dict(group=entry['group'],**cell,arm=arm,gpu=entry['gpu'],ttft_ms=avg(arm),ttft_spread_pct=100*(max(ts)-min(ts))/st.fmean(ts))
            for key in ('mean_tbt_ms','mean_e2e_ms','preemptions','recomputed_tokens','interventions'):
                values=[r['metrics'][key] for r in rs];row[key]=st.fmean(values) if all(v is not None for v in values) else None
            for refarm in ('old','arrival','combined','pressure_constant'):
                row[f'gain_{refarm}_pct']=100*(1-avg(arm)/avg(refarm))
                for rep in (0,1):
                    num=next(r['metrics']['mean_ttft_ms'] for r in rs if r['job']['repeat']==rep)
                    den=next(r['metrics']['mean_ttft_ms'] for r in arms[refarm] if r['job']['repeat']==rep)
                    row[f'gain_{refarm}_r{rep}_pct']=100*(1-num/den)
            pairs.append(row)
        accepted.append(entry['group'])
    missing=sorted(set(cells)-set(accepted))
    if status['state']=='complete':assert not missing and not failures
    summary=dict(updated=__import__('time').time(),gpu_state=status['state'],complete_matrix=not missing,
        accepted_groups=len(accepted),planned_groups=len(cells),formal_runs=len(flat),calibration_runs=2*len(calibrated),
        audit=dict(passed=True,rounds=sum(c['rounds'] for c in checks),arrival_events=sum(c['arrival_events'] for c in checks)),
        missing_groups=missing,failed_groups=failures,state_coverage=coverage,comparisons=pairs,
        simulator=read(OUT/'simulator/full_complete.json') if (OUT/'simulator/full_complete.json').exists() else dict(complete=False))
    aggregates=[]
    for family in sorted({r['family'] for r in pairs}):
        subset=[r for r in pairs if r['family']==family]
        for field in ('family','source','window','shape','capacity','stage'):
            for value in sorted({str(r[field]) for r in subset}):
                for arm in d['methods']:
                    rr=[r for r in subset if str(r[field])==value and r['arm']==arm]
                    aggregates.append(dict(family=family,scope=field,value=value,arm=arm,cells=len(rr),
                        **{k:st.fmean(r[k] for r in rr) for k in ('gain_old_pct','gain_arrival_pct','gain_combined_pct','gain_pressure_constant_pct')},
                        worst_old_pct=min(r['gain_old_pct'] for r in rr),worse_than_old=sum(r['gain_old_pct']<0 for r in rr)))
    summary['aggregates']=aggregates
    for name,rows in [('runs.csv',flat),('comparisons.csv',pairs),('aggregates.csv',aggregates),('audits.csv',[{k:v for k,v in c.items() if not isinstance(v,dict)} for c in checks])]:
        if rows:
            with (OUT/'analysis'/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    write(OUT/'analysis/summary.json',summary);report(summary);plot(pairs)
    print(json.dumps({k:summary[k] for k in ('complete_matrix','accepted_groups','planned_groups','formal_runs','audit','simulator')},indent=2))


def report(s):
    names=dict(old='旧式',arrival='旧+A',combined='原D+A',pressure_decode='压力D+A',pressure_constant='压力常数+A')
    lines=['# 12小时范围扩展：固定公式比较','',
        f"有效完整组 {s['accepted_groups']}/{s['planned_groups']}；正式运行{s['formal_runs']}；校准运行{s['calibration_runs']}。独立审计{s['audit']['rounds']:,}轮通过。",'',
        '**矩阵完整。**' if s['complete_matrix'] else '**矩阵尚不完整：以下仅为已完成完整组，不据此宣布全范围赢家。缺失组、失败和截止情况见summary.json。**','',
        '唯一主指标mean TTFT。以下正值为相对旧式改善；每单元先平均两次，再等权平均相对变化。来源/形状共享请求，不是独立数据集数。所有系数事前固定，没有本轮调参。',
        '所有实际推理使用Qwen2.5-7B；BurstGPT的ChatGPT/GPT-4标签只表示请求长度来源。请求为固定token ID的长度回放。原始记录筛选output≤512且prompt+output≤8000；未裁剪token长度。四种到达形状均受控或缩放，不作为原始自然到达时标实验。','',
        '| 负载类别 | 分组 | 取值 | 方法 | 单元数 | 相对旧式 | 相对A | 相对原组合 | 最差相对旧式 | 退化单元数 |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in s['aggregates']:
        lines.append(f"| {r['family']} | {r['scope']} | {r['value']} | {names[r['arm']]} | {r['cells']} | {r['gain_old_pct']:+.2f}% | {r['gain_arrival_pct']:+.2f}% | {r['gain_combined_pct']:+.2f}% | {r['worst_old_pct']:+.2f}% | {r['worse_than_old']} |")
    lines += ['','## KV状态覆盖（所有正式轨迹逐轮统计）','',str(s['state_coverage']),'',
        '覆盖是各方法实际访问状态的描述，不是固定状态的因果效应。F、P、D占用恒等式已逐轮核对；阈值、到达估计与物理候选由原始记录独立重建。',
        '完整逐场景/重复/抢占/重算/TBT结果见comparisons.csv、runs.csv。GPU/容量校准未通过和外来任务污染不进入有效组；恢复尝试保留在attempts目录。',
        '模拟器使用人工步时验证因果性、可行性和行为覆盖，不把模拟时间当作真实TTFT。数学解释依赖THEORY_CN.md列出的假设，不构成无条件最优证明。','',
        '![压力常数收益](../../artifacts/v1_broad12h_20260925_r1/figures/pressure_constant_map.png)','',
        '[方案](PROTOCOL_CN.md) · [数学解释](THEORY_CN.md) · [进度](PROGRESS_CN.md) · [原始汇总](../../artifacts/v1_broad12h_20260925_r1/analysis/summary.json)']
    (HERE/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')


def plot(rows):
    if not rows:return
    os.environ.setdefault('MPLCONFIGDIR','/tmp/broad12h-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    selected=[r for r in rows if r['arm']=='pressure_constant' and r['family']=='measured_lengths']
    labels=sorted({r['source']+'/'+r['shape'] for r in selected});caps=[512,768,1024,1536,2048,3072,4096]
    if not labels:return
    z=np.full((len(labels),len(caps)),np.nan)
    for i,label in enumerate(labels):
        for j,cap in enumerate(caps):
            vals=[r['gain_old_pct'] for r in selected if r['source']+'/'+r['shape']==label and r['capacity']==cap]
            if vals:z[i,j]=st.fmean(vals)
    fig,ax=plt.subplots(figsize=(10,max(4,len(labels)*.38)))
    limit=max(5,float(np.nanmax(np.abs(z))));im=ax.imshow(z,cmap='RdYlGn',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(range(len(caps)),caps);ax.set_yticks(range(len(labels)),labels)
    for i in range(len(labels)):
        for j in range(len(caps)):
            if not np.isnan(z[i,j]):ax.text(j,i,f'{z[i,j]:+.1f}',ha='center',va='center',fontsize=8)
    ax.set_title('Pressure constant vs old: mean TTFT gain (%)\nCompleted paired cells only; windows averaged where present')
    fig.colorbar(im,ax=ax);fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(OUT/f'figures/pressure_constant_map.{ext}',dpi=160)
    plt.close(fig)

if __name__=='__main__':main('--partial' in sys.argv)
