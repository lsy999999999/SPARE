"""Independently recompute the completed capacity study from saved run JSON."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'artifacts/v1_capacity_gpu_20260923_r1'
OUT=ROOT/'research/v1_capacity_ablation_20260923'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(rows,key): return st.fmean(r['metrics'][key] for r in rows)


def main():
    manifest=json.loads((RUN/'manifest.json').read_text())
    summary=json.loads((RUN/'summary.json').read_text())
    selection=json.loads((RUN/'selection.json').read_text())
    rows=[]
    for path in sorted((RUN/'runs').glob('*.json')):
        row=json.loads(path.read_text())
        row['_path']=path.name
        rows.append(row)
    assert len(rows)==999 and summary['runs']==999 and summary['complete'] and summary['all_runs_successful']
    assert all(r['status']=='complete' and r['completed']==r['request_count'] for r in rows)
    counts=Counter(r['job']['phase'] for r in rows)
    assert counts=={'compare':840,'tune':96,'aa':36,'smoke':27}
    assert len({r['job']['key'] for r in rows})==999
    hashes={line.split()[1]:line.split()[0] for line in (RUN/'run_hashes.sha256').read_text().splitlines()}
    assert len(hashes)==999
    for r in rows:
        relative='artifacts/v1_capacity_gpu_20260923_r1/runs/'+r['_path']
        assert hashes[relative]==digest(RUN/'runs'/r['_path'])
        assert r['job']['gpu_uuid']==manifest['hardware'][str(r['job']['gpu'])]['uuid']
        assert len(r['requests'])==len(r['clocks'])==r['request_count']
        ttft=[]; e2e=[]
        for req in r['requests']:
            clock=r['clocks'][req['id']]
            assert clock['tokens']==req['output']==len(clock['token_times_ms'])
            assert clock['first_ms']==clock['token_times_ms'][0]
            ttft.append(clock['first_ms']-clock['arrival_ms'])
            e2e.append(clock['finish_ms']-clock['arrival_ms'])
        assert math.isclose(st.fmean(ttft),r['metrics']['mean_ttft_ms'],abs_tol=1e-6)
        assert math.isclose(st.fmean(e2e),r['metrics']['mean_e2e_ms'],abs_tol=1e-6)
        assert math.isclose(sum(c['tokens'] for c in r['clocks'].values())/(r['elapsed_ms']/1000),
                            r['metrics']['throughput_tokens_s'],abs_tol=1e-9)
    compare=[r for r in rows if r['job']['phase']=='compare']
    cells=defaultdict(list)
    for r in compare:
        j=r['job']; cells[(j['capacity'],j['arrival'],j['block'],j['method'])].append(r)
    for key,rs in cells.items():
        assert len(rs)==2 and {r['job']['repeat'] for r in rs}=={0,1},key
        request_signatures={(tuple((q['id'],q['prompt'],q['output'],q['arrival_ms']) for q in r['requests'])) for r in rs}
        assert len(request_signatures)==1,key
    core=('a_frozen','fixed_small','occupancy','srp_matched')
    capacities=(768,1024,1536,2048,4096,7445)
    arrivals=('original','burst20ms')
    table=[]; blocks=[]
    for cap in capacities:
        for arrival in arrivals:
            block_names=sorted({r['job']['block'] for r in compare if r['job']['capacity']==cap and
                                r['job']['arrival']==arrival and r['job']['method']=='srp_matched'})
            assert len(block_names)==6
            block_values={}
            for block in block_names:
                request_sets=[]
                for method in core:
                    rs=cells[(cap,arrival,block,method)]
                    request_sets.append(tuple(q['id'] for q in rs[0]['requests']))
                    block_values[(block,method)]={k:mean(rs,k) for k in (
                        'mean_ttft_ms','p95_ttft_ms','mean_e2e_ms','throughput_tokens_s',
                        'preemptions','recomputed_tokens','interventions','decision_ms_total')}
                assert len(set(request_sets))==1
            baseline=st.fmean(block_values[(b,'srp_matched')]['mean_ttft_ms'] for b in block_names)
            for method in core:
                ttft=st.fmean(block_values[(b,method)]['mean_ttft_ms'] for b in block_names)
                item=dict(capacity=cap,arrival=arrival,method=method,blocks=6,
                    mean_ttft_ms=ttft,gain_vs_srp_pct=(baseline-ttft)/baseline*100,
                    block_wins_vs_srp=sum(block_values[(b,method)]['mean_ttft_ms']<
                        block_values[(b,'srp_matched')]['mean_ttft_ms'] for b in block_names),
                    p95_ttft_ms=st.fmean(block_values[(b,method)]['p95_ttft_ms'] for b in block_names),
                    mean_e2e_ms=st.fmean(block_values[(b,method)]['mean_e2e_ms'] for b in block_names),
                    throughput_tokens_s=st.fmean(block_values[(b,method)]['throughput_tokens_s'] for b in block_names),
                    preemptions=st.fmean(block_values[(b,method)]['preemptions'] for b in block_names),
                    recomputed_tokens=st.fmean(block_values[(b,method)]['recomputed_tokens'] for b in block_names),
                    interventions=st.fmean(block_values[(b,method)]['interventions'] for b in block_names),
                    decision_ms_total=st.fmean(block_values[(b,method)]['decision_ms_total'] for b in block_names))
                table.append(item)
            for block in block_names:
                base=block_values[(block,'srp_matched')]['mean_ttft_ms']
                blocks.append(dict(capacity=cap,arrival=arrival,block=block,
                    source=block.split('_')[0],srp_ttft_ms=base,**{
                    m+'_gain_pct':(base-block_values[(block,m)]['mean_ttft_ms'])/base*100
                    for m in core[:-1]}))
    ablations=[]
    for cap in (768,1536,2048):
        for arrival in arrivals:
            block_names=sorted({r['job']['block'] for r in compare if r['job']['capacity']==cap and
                                r['job']['arrival']==arrival and r['job']['method']=='srp_matched'})
            for method in ('a_any_positive_saving','a_frozen','a_no_finish_guard',
                           'occupancy_finish_guard','occupancy','fixed_small','srp_matched'):
                values=[]
                for block in block_names:
                    rs=cells[(cap,arrival,block,method)]
                    values.append({k:mean(rs,k) for k in ('mean_ttft_ms','throughput_tokens_s',
                                  'preemptions','recomputed_tokens','interventions','rejected_guard')})
                base=next(x for x in table if x['capacity']==cap and x['arrival']==arrival and x['method']=='srp_matched')['mean_ttft_ms']
                ttft=st.fmean(v['mean_ttft_ms'] for v in values)
                ablations.append(dict(capacity=cap,arrival=arrival,method=method,
                    mean_ttft_ms=ttft,gain_vs_srp_pct=(base-ttft)/base*100,
                    throughput_tokens_s=st.fmean(v['throughput_tokens_s'] for v in values),
                    preemptions=st.fmean(v['preemptions'] for v in values),
                    recomputed_tokens=st.fmean(v['recomputed_tokens'] for v in values),
                    interventions=st.fmean(v['interventions'] for v in values),
                    rejected_guard=st.fmean(v['rejected_guard'] for v in values)))
    shadow=[]
    for cap in (768,7445):
        for arrival in arrivals:
            item={}
            for method in ('a_shadow','srp_matched'):
                rs=[r for r in compare if r['job']['capacity']==cap and r['job']['arrival']==arrival and r['job']['method']==method]
                item[method]=dict(ttft=mean(rs,'mean_ttft_ms'),decision_ms_total=mean(rs,'decision_ms_total'))
            shadow.append(dict(capacity=cap,arrival=arrival,**item))
    payload=dict(complete=True,run_hashes_verified=999,selection=selection,
        counts=dict(counts),core=table,blocks=blocks,ablations=ablations,shadow=shadow,
        caveats=['Six independent request windows; repeats are technical repeats.',
                 'Original arrivals cover only the first 64 requests of each predeclared minute.',
                 'These are length replays with synthetic prompt token IDs, not semantic prompt replay.'])
    (OUT/'RESULTS_AUDIT.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    lines=['# 新负载容量曲线：独立重算快报','',
        '999/999 次运行完整成功；999 份保存记录哈希通过，逐请求 TTFT/E2E、吞吐和配对输入已从原始 JSON 重算。',
        f"开发集选择：固定预算 {selection['fixed_small_budget']}，简单 KV 占用阈值 {selection['occupancy_threshold']:.0%}。",'',
        '## 核心容量曲线','',
        '| KV blocks | 到达 | 方法 | mean TTFT ms | 对同配 SRP 改善 | 获益窗口 | 吞吐 tok/s | 抢占/运行 | 干预/运行 |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    labels={'a_frozen':'A','fixed_small':'固定512','occupancy':'占用率70%','srp_matched':'SRP'}
    for x in table:
        lines.append(f"| {x['capacity']} | {x['arrival']} | {labels[x['method']]} | {x['mean_ttft_ms']:.1f} | {x['gain_vs_srp_pct']:+.1f}% | {x['block_wins_vs_srp']}/6 | {x['throughput_tokens_s']:.1f} | {x['preemptions']:.1f} | {x['interventions']:.1f} |")
    lines += ['','## 关键消融','',
        '| KV blocks | 到达 | 方法 | mean TTFT ms | 对 SRP 改善 | 吞吐 tok/s | 抢占/运行 | 干预/运行 |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for x in ablations:
        if x['method'] not in ('a_any_positive_saving','a_frozen','a_no_finish_guard','occupancy_finish_guard'): continue
        lines.append(f"| {x['capacity']} | {x['arrival']} | {x['method']} | {x['mean_ttft_ms']:.1f} | {x['gain_vs_srp_pct']:+.1f}% | {x['throughput_tokens_s']:.1f} | {x['preemptions']:.1f} | {x['interventions']:.1f} |")
    lines += ['','## 当前读法','',
        '- A 在 768、1024 blocks 的两种到达模式均明显胜过同配置 SRP；1536 仍获益，但简单基线已追平或略好。',
        '- 2048 blocks 的冻结 A 几乎等同 SRP；4096、7445 不干预。当前 7/256 总容量门槛限制了适用区间。',
        '- 1-block 节约门槛消融在 768、1536、2048 都强，尤其在 2048 恢复了吞吐并减少抢占。这是下一版规则的开发线索，不是可直接作为冻结新方法的确认结果。',
        '- 去掉本轮完成保护的均值与 A 很接近，当前六窗口不足以证明该保护带来性能收益；仍需检查逐请求坏例与公平性。',
        '- 固定 512 和 70% 占用率规则在 1536/2048 有竞争力，说明成熟论文必须正面对照简单控制，不能只对 SRP。',
        '', '技术重复先在窗口内平均，获益窗口分母为 6；请求和重复没有当作独立样本。本文是完成结果的快报，尚未做完整逐轮 trace 审计。']
    (OUT/'RESULTS_AUDIT_CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='passed',runs=999,compare=840,core_rows=len(table),
                          ablation_rows=len(ablations),block_rows=len(blocks)),indent=2))


if __name__=='__main__': main()
