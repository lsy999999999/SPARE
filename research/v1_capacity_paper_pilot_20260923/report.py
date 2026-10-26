"""Independently reconstruct saved clocks and token work, then plot all arms."""
from collections import defaultdict
import csv
import gzip
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.v1_capacity_paper_pilot_20260923.common import *
from research.v1_budget_20260922 import run as ref
ref.OUT = OUT

LABELS = dict(srp='SRP', one_block='1-block', fixed512='Fixed 512', pressure1_8='Pressure 1/8',
    sarathi_common='Sarathi / common', past_future_common='Past-Future / common',
    sarathi_frozen='Sarathi / frozen', past_future_frozen='Past-Future / frozen')


def audit(row):
    ref.audit_result(row)
    with gzip.open(OUT/row['rounds_path'], 'rt') as f: rounds = json.load(f)
    work, high = defaultdict(int), defaultdict(int)
    repeated = 0
    for r in rounds:
        per_round = 0
        assert len(r['allocations']) == len(r['scheduled'])
        for a in r['allocations']:
            rid, before, n = a['id'], a['before'], a['tokens']
            assert r['scheduled'][rid] == n and n > 0
            assert a['blocks_after'] == (before+n+15)//16
            work[rid] += n
            per_round += min(n, max(0, high[rid]-before))
            high[rid] = max(high[rid], before+n)
        assert per_round == r['recomputed_tokens']
        repeated += per_round
    useful = 0
    for req in row['requests']:
        target = req['prompt']+req['output']-1
        assert high[req['id']] == target
        useful += target
    assert sum(work.values()) == useful+repeated
    assert repeated == row['metrics']['recomputed_tokens']
    cs = list(row['clocks'].values())
    e2e = st.fmean(c['finish_ms']-c['arrival_ms'] for c in cs)
    assert math.isclose(e2e, row['metrics']['mean_e2e_ms'], abs_tol=1e-7)
    tbt = [b-a for c in cs for a,b in zip(c['token_times_ms'], c['token_times_ms'][1:])]
    assert math.isclose(ref.quantile(tbt, .95), row['metrics']['p95_tbt_ms'], abs_tol=1e-7)
    interventions = [r for r in rounds if r['intervened']]
    eligible = [r for r in rounds if r.get('baseline_growth_blocks') is not None]
    return dict(useful_tokens=useful, recomputed_tokens=repeated, scheduled_tokens=sum(work.values()),
        interventions=len(interventions),
        interventions_below_pressure1_8=sum(8*r['baseline_growth_blocks'] < max(r['free_before'],1)
            for r in interventions if r.get('baseline_growth_blocks') is not None),
        mean_intervention_free_blocks=st.fmean(r['free_before'] for r in interventions) if interventions else None,
        eligible_baseline_rounds=len(eligible),
        eligible_below_pressure1_8=sum(8*r['baseline_growth_blocks'] < max(r['free_before'],1) for r in eligible))


def main():
    assert read(OUT/'status.json')['state'] == 'finished'
    manifest = read(OUT/'manifest.json')
    assert manifest['sources'] == sources()
    assert manifest['history_sha256'] == digest(HISTORY)
    assert all(digest(INPUT/n) == sha for n,sha in manifest['inputs'].items())
    rows = [read(p) for p in sorted((OUT/'runs').glob('*.json'))]
    assert len(rows) == 108
    counts = defaultdict(int)
    checked = []
    groups = defaultdict(list)
    expected = {(c['key'],phase,arm,r) for c in manifest['cells']
                for phase,arms in (('aa',['srp']),('compare',list(ARMS))) for arm in arms for r in range(2)}
    seen = set()
    for row in rows:
        j = row['job']; cell = j['cell']
        ident = (cell['key'],j['phase'],j['arm'],j['repeat'])
        assert ident in expected and ident not in seen
        seen.add(ident)
        assert cell in manifest['cells']
        method,cfg = ARMS[j['arm']]
        assert j['method'] == method and j['config'] == cfg
        trace = 'code_calibration' if j['phase'] == 'aa' else cell['block']
        assert row['requests'] == read(INPUT/(trace+'_original.json'))
        checked.append(dict(key=j['key'], arm=j['arm'], phase=j['phase'],
            block=cell['block'], capacity=cell['capacity'], **audit(row)))
        counts[j['phase']] += 1
        if j['phase'] == 'compare': groups[(cell['block'],cell['capacity'],j['arm'])].append(row)
    assert seen == expected
    aa_ranges = {}
    for cell in manifest['cells']:
        assert read(OUT/'cells'/(cell['key']+'.json'))['state'] == 'complete'
        rr = [r for r in rows if r['job']['cell']['key']==cell['key'] and r['job']['phase']=='aa']
        ratio = {}
        for name, xs in (('ttft',[r['metrics']['mean_ttft_ms'] for r in rr]),('elapsed',[r['elapsed_ms'] for r in rr])):
            ratio[name] = (max(xs)-min(xs))/st.fmean(xs)
        assert max(ratio.values()) <= .05
        aa_ranges[cell['key']] = ratio
    result = []
    metrics = ('mean_ttft_ms','mean_e2e_ms','p95_tbt_ms','preemptions','recomputed_tokens',
        'interventions','decision_ms_mean','decision_ms_p95','gate_rejected',
        'peak_kv_fraction','low_free_round_fraction','round_count','deferred_prefill_tokens')
    for (block,cap,arm), rr in groups.items():
        assert len(rr) == 2
        xs = [r['metrics']['mean_ttft_ms'] for r in rr]
        result.append(dict(block=block,capacity=cap,arm=arm,
            **{k:st.fmean(r['metrics'][k] for r in rr) for k in metrics},
            ttft_min=min(xs),ttft_max=max(xs),ttft_repeat_range_fraction=(max(xs)-min(xs))/st.fmean(xs)))
    write(OUT/'audit.json', dict(passed=True, counts=dict(counts), aa_ranges=aa_ranges,
        reconstruction='request clocks, exact matrix, source/input hashes, physical blocks and independently reconstructed repeated token work', rows=checked))
    write(OUT/'summary.json', dict(manifest=manifest, aggregates=result))
    with (OUT/'comparison.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(result[0])); w.writeheader(); w.writerows(result)
    lookup = {(r['block'],r['capacity'],r['arm']):r for r in result}
    table = ['# GPU 容量与论文规则比较', '', '全部数字为两次重复平均；TTFT / E2E / TBT 单位 ms。',
        '同一个点在同一张 RTX4090 上运行；八组×两次×六点=96 次比较，另12次 A/A。',
        '这是两个复用开发窗口的探索性比较。论文组为公共 vLLM V1 的规则适配。', '',
        '| 窗口 | KV blocks | 方法 | mean TTFT | mean E2E | p95 TBT | 抢占 | 重算 tokens | 干预 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for b in BLOCKS:
        for c in CAPS:
            for a in ARMS:
                r=lookup[b,c,a]
                table.append(f"| {b} | {c} | {LABELS[a]} | {r['mean_ttft_ms']:.1f} | {r['mean_e2e_ms']:.1f} | {r['p95_tbt_ms']:.1f} | {r['preemptions']:.1f} | {r['recomputed_tokens']:.0f} | {r['interventions']:.0f} |")
    table += ['', 'A/A 最大 range/mean：'+f"{100*max(v for x in aa_ranges.values() for v in x.values()):.2f}%", '',
              '## 每点最小 mean TTFT（描述性排序，不是显著性检验）', '', '| 窗口 | blocks | 最优组 | TTFT | 1-block | fixed512 | pressure1/8 |', '|---|---:|---|---:|---:|---:|---:|']
    for b in BLOCKS:
        for c in CAPS:
            best = min((lookup[b,c,a] for a in ARMS),key=lambda r:r['mean_ttft_ms'])
            values = ' | '.join(f"{lookup[b,c,a]['mean_ttft_ms']:.1f}" for a in ('one_block','fixed512','pressure1_8'))
            table.append(f"| {b} | {c} | {LABELS[best['arm']]} | {best['mean_ttft_ms']:.1f} | {values} |")
    table += ['', 'common = 1024/256/32；frozen = 上一轮 queue 冻结的1024/1024/32。',
        'Past-Future 使用训练历史的输出长度条件采样；其他组不预测输出长度。',
        '图片误差棒为两次重复的 min–max，不是置信区间。所有方法在线审计开销包含在 GPU 计时中。',
        'CSV 中的 p95_tbt_ms 是两次运行 p95 的平均，不是合并样本 p95。',
        '真实 GPU 执行长度回放（prompt 为重复 token，固定实际输出长度），不是语义质量实验。',
        '两个窗口既有长度组成差异，也有自然到达速度差异，不能把窗口差异归因于单一因素。', '']
    (OUT/'RESULTS_CN.md').write_text('\n'.join(table))
    plot(lookup)
    legacy_report()
    print(json.dumps(dict(audit_passed=True,counts=dict(counts),max_repeat_range=max(r['ttft_repeat_range_fraction'] for r in result))))


def plot(lookup):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/v1-capacity-paper-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#7c8794','#2563eb','#f59e0b','#10a37f','#a855f7','#db2777','#7c3aed','#be123c']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    dest=OUT/'figures'; dest.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(16,9),sharey=True)
    for i,b in enumerate(BLOCKS):
        for j,c in enumerate(CAPS):
            ax=axes[i,j]
            rr=[lookup[b,c,a] for a in ARMS]
            means=[r['mean_ttft_ms']/1000 for r in rr]
            errors=[[ (r['mean_ttft_ms']-r['ttft_min'])/1000 for r in rr],
                    [(r['ttft_max']-r['mean_ttft_ms'])/1000 for r in rr]]
            ax.barh(range(8),means,color=colors,xerr=errors,error_kw={'capsize':2,'lw':1})
            ax.set_yticks(range(8),[LABELS[a] for a in ARMS]); ax.set_ylim(7.5,-.5)
            ax.set_title(f'{b.split("_")[0].capitalize()} | {c} KV blocks')
            ax.set_xlabel('Mean TTFT (s), lower is better')
            ax.set_xlim(0,max(means)*1.22); ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
            for y,v in enumerate(means): ax.text(v+max(means)*.015,y,f'{v:.2f}',va='center',fontsize=9)
    fig.suptitle('Capacity pilot: simple rules and paper-rule adaptations',fontsize=16,y=.99)
    fig.text(.5,.01,'Qwen2.5-7B / RTX4090 / vLLM V1 | 64 requests per reused window | 2 repeats, bars show min-max | No new tuning',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.96))
    for suffix in ('png','pdf'): fig.savefig(dest/f'ttft_all_methods.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    chosen=('srp','one_block','fixed512','pressure1_8','past_future_frozen')
    for ax,b in zip(axes,BLOCKS):
        for a in chosen:
            vals=[lookup[b,c,a]['mean_ttft_ms']/lookup[b,c,'srp']['mean_ttft_ms'] for c in CAPS]
            ax.plot(CAPS,vals,marker='o',label=LABELS[a],color=colors[list(ARMS).index(a)])
        ax.set_title(b.split('_')[0].capitalize()); ax.set_xlabel('KV capacity (blocks)')
        ax.set_xticks(CAPS); ax.grid(alpha=.2); ax.axhline(1,color='#999',lw=.6)
    axes[0].set_ylabel('Mean TTFT / SRP (lower is better)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False)
    fig.suptitle('The benefit of aggressive KV saving changes with capacity')
    fig.tight_layout(rect=(0,.09,1,.95))
    for suffix in ('png','pdf'): fig.savefig(dest/f'capacity_trend.{suffix}',dpi=180)
    plt.close(fig)


def legacy_report():
    dest=OUT/'legacy_simulator_r2'
    manifest=read(dest/'manifest.json')
    assert all(digest(ROOT/p)==v for p,v in manifest['sources'].items())
    assert all(digest(ROOT/p)==v for p,v in manifest['training'].items())
    assert all(digest(dest/'inputs'/(b+'.json'))==v for b,v in manifest['inputs'].items())
    rows=read(dest/'summary.json'); assert len(rows)==36
    for b in BLOCKS:
        a=read(dest/'inputs'/(b+'.json')); original=read(INPUT/(b+'_original.json'))
        assert len(a)==len(original)==64
        for i,(x,y) in enumerate(zip(a,original)):
            assert (x['request_id'],x['original_id'],x['arrival_ms'],x['prompt_tokens'],x['actual_output_tokens']) == (i,y['id'],y['arrival_ms'],y['prompt'],y['output'])
    for r in rows:
        assert r['metrics']['completed']==r['metrics']['requests']==64
        assert not r['identity']['uses_hidden_output_for_scheduling']
        assert r['validation']['accuracy'] is None
        with (dest/r['name']/'requests.csv').open() as f: requests=list(csv.DictReader(f))
        assert len(requests)==64 and len({x['request_id'] for x in requests})==64
        assert all(x['complete']=='True' for x in requests)
        for k in ('ttft_ms','e2e_ms'):
            assert math.isclose(st.fmean(float(x[k]) for x in requests),r['metrics'][k+'_mean'],abs_tol=1e-6)
        assert sum(int(x['evictions']) for x in requests)==r['metrics']['evictions']
    write(dest/'audit.json',dict(passed=True,runs=36,requests=36*64,
        checks='request clocks, completion, source/training/input hashes, exact GPU input lengths/arrivals, no oracle arms',
        limitation='No matched GPU calibration. This is not native V1 or full paper-system reproduction.'))
    os.environ.setdefault('MPLCONFIGDIR','/tmp/v1-capacity-paper-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    byname={r['name']:r['metrics'] for r in rows}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    policies=('fcfs','orca-max','s3','ltr','mlfq','sarathi')
    for ax,b in zip(axes,BLOCKS):
        for p in policies:
            values=[byname[f'{b}_k{c}_{p}']['ttft_ms_mean']/byname[f'{b}_k{c}_fcfs']['ttft_ms_mean'] for c in CAPS]
            ax.plot(CAPS,values,marker='o',label=p)
        ax.set_title(b.split('_')[0]); ax.set_xlabel('KV capacity (blocks)'); ax.set_xticks(CAPS); ax.grid(alpha=.2)
    axes[0].set_ylabel('Simulated mean TTFT / FCFS')
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=6,frameon=False)
    fig.suptitle('Supplement: uncalibrated analytic simulator, paper rules only')
    fig.tight_layout(rect=(0,.08,1,.95))
    fig.savefig(dest/'simulator_only.png',dpi=180); fig.savefig(dest/'simulator_only.pdf')
    plt.close(fig)


if __name__ == '__main__': main()
