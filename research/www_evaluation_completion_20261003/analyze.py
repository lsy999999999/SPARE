"""Summaries preserve profiles, controls, adverse outcomes and incomplete cells."""
import csv
import statistics
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from .common import OUT, ROOT, GPU, HERE, read, write


def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze():
    design=read(OUT/'design.json');results=[];missing=[]
    for t in design['tasks']:
        p=OUT/'runs'/t['name']/'result.json'
        if not p.exists() or not read(p).get('full_source_sha256'):
            missing.append(t['name']);continue
        results.append(read(p))
    flat=[dict(name=r['task']['name'],family=r['task']['family'],profile=r['profile'],
               case=r['task']['case'],window=r['task'].get('window','gpu_fresh1'),capacity=r['task']['capacity'],
               arm=r['arm'],arrival_rps=r['task'].get('arrival_rps'),**r['metrics']) for r in results]
    keys=sorted({k for r in flat for k in r})
    csv_write(OUT/'metrics.csv',[{k:r.get(k) for k in keys} for r in flat])
    index={(r['profile'],r['task']['case'],r['task']['capacity'],r['arm']):r for r in results}
    pairs=[]
    for r in results:
        if r['task']['family']=='validation' or r['arm']=='spare':continue
        base=index.get((r['profile'],r['task']['case'],r['task']['capacity'],'spare'))
        if base is None:continue
        b=r['metrics']['ttft_ms_mean'];a=base['metrics']['ttft_ms_mean']
        pairs.append(dict(family=r['task']['family'],profile=r['profile'],case=r['task']['case'],
                          window=r['task']['window'],capacity=r['task']['capacity'],control=r['arm'],
                          spare_ttft_ms=a,control_ttft_ms=b,spare_reduction_pct=100*(b-a)/b,
                          trajectory_identical=r['trajectory_sha256']==base['trajectory_sha256'],
                          finish_opportunities_deferred=r['metrics'].get('finish_opportunities_deferred',0)))
    csv_write(OUT/'paired_comparisons.csv',pairs)
    grouped=defaultdict(list)
    for p in pairs:grouped[p['family'],p['profile'],p['control']].append(p)
    summary=[];rng=np.random.default_rng(20261003)
    for (family,profile,arm),items in sorted(grouped.items()):
        windows=defaultdict(list)
        for p in items:windows[p['window']].append(p['spare_reduction_pct'])
        values=np.array([statistics.fmean(v) for v in windows.values()])
        bootstrap=rng.choice(values,size=(5000,len(values)),replace=True).mean(axis=1)
        summary.append(dict(family=family,profile=profile,control=arm,cells=len(items),windows=len(values),
                            mean_reduction_pct=float(np.mean(values)),ci_low=float(np.quantile(bootstrap,.025)),
                            ci_high=float(np.quantile(bootstrap,.975)),
                            wins=sum(p['spare_reduction_pct']>1e-9 for p in items),
                            ties=sum(abs(p['spare_reduction_pct'])<=1e-9 for p in items),
                            losses=sum(p['spare_reduction_pct']< -1e-9 for p in items),
                            same_trajectories=sum(p['trajectory_identical'] for p in items)))
    csv_write(OUT/'comparison_summary.csv',summary)
    # Compare 4-repeat GPU averages against simulated requests, matched by identity.
    runs=list(csv.DictReader((GPU/'analysis/runs.csv').open()));gpu=defaultdict(list)
    for run in runs:
        if 'fresh1' in run['cell']:
            gpu[Path(run['source_input']).stem,int(run['capacity']),run['arm']].append(run)
    validation=[];request_errors=[];direction=[]
    for r in results:
        t=r['task']
        if t['family']!='validation':continue
        name=Path(t['input']).stem;matched=gpu[name,t['capacity'],r['arm']]
        assert len(matched)==4
        sim=list(csv.DictReader((OUT/'runs'/t['name']/'requests.csv').open()))
        actual=defaultdict(list)
        for run in matched:
            for req in read(GPU/run['result_path'])['requests']:actual[req['request_id']].append(req['ttft_ms'])
        rel=[];delta=[]
        for req in sim:
            truth=statistics.fmean(actual[req['request_id']]);pred=float(req['ttft_ms'])
            rel.append(abs(pred-truth)/truth);delta.append(pred-truth)
            request_errors.append(dict(profile=r['profile'],case=t['case'],arm=r['arm'],
                                       request_id=req['request_id'],gpu_ttft_ms=truth,sim_ttft_ms=pred))
        validation.append(dict(profile=r['profile'],case=t['case'],arm=r['arm'],
                               request_mae_ms=statistics.fmean(map(abs,delta)),request_mape_pct=100*statistics.fmean(rel),
                               gpu_ttft_ms=statistics.fmean(float(x['ttft_ms_mean']) for x in matched),
                               sim_ttft_ms=r['metrics']['ttft_ms_mean']))
    csv_write(OUT/'gpu_validation.csv',validation);csv_write(OUT/'gpu_request_errors.csv',request_errors)
    vidx={(r['profile'],r['case'],r['arm']):r for r in validation}
    for v in validation:
        if v['arm']=='arrival':continue
        a=vidx.get((v['profile'],v['case'],'arrival'))
        if not a:continue
        g=100*(v['gpu_ttft_ms']-a['gpu_ttft_ms'])/v['gpu_ttft_ms']
        s=100*(v['sim_ttft_ms']-a['sim_ttft_ms'])/v['sim_ttft_ms']
        direction.append(dict(profile=v['profile'],case=v['case'],control=v['arm'],gpu_reduction_pct=g,
                               sim_reduction_pct=s,same_sign=(g>0)==(s>0),gpu_abs_effect_above_2pct=abs(g)>2))
    csv_write(OUT/'gpu_gain_directions.csv',direction)
    write(OUT/'summary.json',dict(expected=design['expected'],completed=len(results),missing=missing,
          by_family=dict(Counter(r['task']['family'] for r in results)),comparisons=summary,
          performance_validated=False,
          uncertainty='Exploratory bootstrap over six source/date windows; deterministic profiles and repeated shapes are not independent samples.'))
    lines=['# SPARE Evaluation 补齐实验（2026-10-03）','',
           f'完成 {len(results)}/{design["expected"]} 条原生 V1 CPU 模拟轨迹；缺失 {len(missing)} 条。未启动 GPU 推理。','',
           '本批覆盖当前版本的简单控制、首 token 完成保护、参数敏感性、持续负载，以及与归档 RTX 4090 结果的逐请求比较。',
           '两套原有合成时间假设与 GPU 历史轨迹拟合时间分开报告。拟合仅使用 fresh0；验证使用不重叠的 fresh1。它是回顾性校准，不是新采集的独立 profiling，也不自动证明硬件预测有效。','',
           '新负载来自两个 Azure 来源、各三个预定日期窗口，共 3072 个未出现在扫描记录中的请求身份。每个窗口前128请求用于主消融/参数实验，全部512请求用于2/5/10/20 req/s持续负载；长度不裁剪，输出≤512且总长≤8000。容量为768/2048/4096配置块，可用块减1。','',
           '持续负载仍是有限窗口，运行完最后一次到达后排空；不得把请求吞吐写成已测最大稳定容量。三个profile共用负载，不合并成三倍独立样本。','',
           '## 完整方法相对各对照','',
           '正数表示 SPARE 更快。先在每个源窗口内平均配对相对降低，再平均六窗口；区间为六窗口成组重采样的探索性95%区间。包含全部容量、形状与负收益。','',
           '| 实验 | 时间模型 | 对照 | 单元数 | SPARE降低TTFT | 95%区间 | 胜/平/负 |',
           '|---|---|---|---:|---:|---|---|']
    for r in summary:
        lines.append(f'| {r["family"]} | {r["profile"]} | {r["control"]} | {r["cells"]} | {r["mean_reduction_pct"]:+.2f}% | [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}] | {r["wins"]}/{r["ties"]}/{r["losses"]} |')
    lines+=['','## GPU留出窗口诊断','','| 时间模型 | 验证单元 | 场景平均逐请求TTFT MAPE | GPU效应超过2%的对照中方向一致 |','|---|---:|---:|---:|']
    for profile in design['profiles']:
        group=[v for v in validation if v['profile']==profile]
        dd=[d for d in direction if d['profile']==profile and d['gpu_abs_effect_above_2pct']]
        if group:lines.append(f'| {profile} | {len(group)} | {statistics.fmean(v["request_mape_pct"] for v in group):.2f}% | {sum(d["same_sign"] for d in dd)}/{len(dd)} |')
    lines+=['','2%仅为描述性筛选，用于避免接近零的GPU差异主导符号计数，不是显著性检验。TRAIL与UniBoost因GPU/CPU抢占适配不同排除于本验证；既有13方法GPU主表完整保留。','',
            '## 开销与保护检查','']
    if (OUT/'overhead.json').exists():
        o=read(OUT/'overhead.json');lines+=[f'检查 {o["sampled_states"]} 个归档GPU状态，提前停止与全候选审计的动作差异 {o["early_exit_action_mismatches"]}；去掉decode保护改变 {o["removing_decode_guard_changed_actions"]} 个动作；去掉首token完成保护改变 {o["removing_finish_guard_changed_actions"]} 个动作。',
              'CPU策略微基准不含排序、状态提取、原生调度或GPU时间。完整测量范围见 overhead.json。']
        for k,v in o['costs'].items():lines.append(f'- {k}: 平均 {v["mean_us"]:.2f} us，P95 {v["p95_us"]:.2f} us。')
    else:lines+=['尚未完成独立CPU策略微基准。']
    lines+=['','## 写作材料与复现','',
            '- metrics.csv：全指标、到达四分位TTFT、排空时长、峰值积压、保护与干预统计。',
            '- paired_comparisons.csv / comparison_summary.csv：逐场景配对与按窗口汇总。',
            '- gpu_validation.csv / gpu_gain_directions.csv：绝对误差与相对收益方向。',
            '- runs/：逐请求结果、压缩逐轮记录；design.json与source/固定输入和代码。',
            '- failures.json：失败保留，不插值、不按胜负重跑。',
            '- 主GPU矩阵仍使用 writing/www/experiments/gpu_baselines_4090_20260930_r3；本批补充结果不能改名成GPU收益。','']
    (OUT/'REPORT_CN.md').write_text('\n'.join(lines))
    (HERE/'RESULTS_CN.md').write_text('# Evaluation 补齐结果\n\n[完整报告](../../artifacts/www_evaluation_completion_20261003_r1/REPORT_CN.md)\n')
    import subprocess, os
    subprocess.run([str(ROOT/"simulator/bin/python"), "-B", "-m", "research.www_evaluation_completion_20261003.plot"], cwd=ROOT, env={**os.environ,"MPLCONFIGDIR":str(OUT/"matplotlib_cache")}, check=True)


if __name__=='__main__':analyze()
