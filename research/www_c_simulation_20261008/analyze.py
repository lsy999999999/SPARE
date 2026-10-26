"""Independently reaggregate the frozen C simulation and write a review."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import statistics as st
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.www_c_simulation_20261008.run import GPU, OUT, OLD, read, write, digest


def main():
    design=read(OUT/'design.json'); summary=read(OUT/'summary.json')
    for name,h in design['source_sha256'].items():assert digest(ROOT/name)==h,name
    assert digest(OLD/'timing_profile.json')==design['timing_profile_sha256']
    assert digest(GPU/'design.json')==design['gpu_design_sha256']
    input_ids=set()
    for c in design['cells']:
        assert digest(GPU/c['input'])==c['input_sha256']
        input_ids.update(r['id'] for r in read(GPU/c['input']))
    assert not input_ids & set(read(OLD/'timing_profile.json')['training_request_ids'])
    raw={}; cal=defaultdict(list)
    for path in sorted((OUT/'runs').glob('*/result.json')):
        r=read(path); t=r['c_task']; c=t['cell']
        assert r['complete'] and r['metrics']['completed']==128
        assert r['c_design_sha256']==digest(OUT/'design.json')
        req=list(csv.DictReader((path.parent/'requests.csv').open()))
        assert len(req)==128 and {x['request_id'] for x in req}=={x['id'] for x in read(GPU/c['input'])}
        for field in ('ttft_ms','e2e_ms'):
            assert abs(st.fmean(float(q[field]) for q in req)-r['metrics'][field+'_mean'])<1e-8
        key=(t['profile'],c['key'],t['arm']);assert key not in raw
        raw[key]=r
        if c['stage']=='cal':cal[t['profile'],t['arm']].append(r['metrics']['ttft_ms_mean'])
    assert len(raw)==summary['total_runs']==672
    assert not list((OUT/'runs').glob('*/error.json'))
    for p,selection in summary['selection'].items():
        for arm,score in selection['scores'].items():assert abs(st.fmean(cal[p,arm])-score)<1e-8
        for family in ('fixed','guarded'):
            assert selection[family]==min(design['budgets'],key=lambda b:(selection['scores'][f'{family}_{b}'],b))
    accepted={r['cell'] for r in csv.DictReader((GPU/'analysis/partial_cell_means.csv').open())}
    for aggregate in summary['summary']:
        p=aggregate['profile'];ref=aggregate['reference'];actual=ref
        if ref.endswith('gpu256') and summary['selection'][p][ref.split('_')[0]]==256:actual=ref.replace('gpu256','selected')
        cells=[c for c in design['cells'] if c['stage']=='test' and (aggregate['scope']=='full24' or c['key'] in accepted)]
        values=[100*(1-raw[p,c['key'],'spare']['metrics']['ttft_ms_mean']/raw[p,c['key'],actual]['metrics']['ttft_ms_mean']) for c in cells]
        assert len(values)==aggregate['n']
        assert abs(st.fmean(values)-aggregate['ttft_reduction_pct'])<1e-9
        assert sum(v>1e-9 for v in values)==aggregate['wins']
    clock=read(OUT/'clock_sensitivity.json');assert clock['complete'] and clock['runs']==25
    for row in clock['rows']:
        r=read(OUT/'clock_sensitivity'/f"{row['scale']:.2f}__{row['arm']}"/'result.json')
        assert r['complete'] and r['metrics']['completed']==128
        assert r['diagnostic_design_sha256']==digest(OUT/'diagnostic_design.json')
        assert abs(row['ttft_ms']-r['metrics']['ttft_ms_mean'])<1e-8
    checks=dict(passed=True,formal_runs=672,clock_diagnostic_runs=25,failures=0,
        source_input_hashes_match=True,training_input_id_overlap=0,raw_request_means_reaggregated=True,
        all_paired_summaries_reaggregated=True,calibration_selection_recomputed=True,
        native_gpu_burst_validation=read(OUT/'validation.json'))
    write(OUT/'checks.json',checks)
    names={'resource':'Resource','resource_staircase':'Staircase','gpu_fitted':'Fitted'}
    gpu=read(GPU/'analysis/partial_summary.json')
    lookup={(r['profile'],r['scope'],r['reference']):r for r in summary['summary']}
    def effect(p,scope,ref):return lookup[p,scope,ref]['ttft_reduction_pct']
    lines=['# C 设计复查与实际模拟结果','',
        '结论：C 的基本对照结构合理，但不能说测量设计和实现完全没有问题。失败场景有清楚的时序敏感性；然而，在GPU同样的23场景上，Fitted对固定256的配对TTFT差异与GPU接近且方向一致，不支持把固定预算对照上的劣势仅归因于GPU波动。','',
        '## 实际执行与语义核验','',
        '- 完成672条正式CPU模拟：288条校准、384条测试（含Resource额外24条GPU固定256对照），零失败。另完成25条固定时钟倍率诊断，零失败。',
        '- 直接复用GPU C的CScheduler/c_policy。旧模拟policy不处理fixed_guarded及candidate_axes，直接传这些参数会失去对照语义；本次通过调用C原实现避免了这个问题，没有修改冻结的GPU代码。',
        '- vLLM 0.9.2调度器及三项KV分配模块的源码哈希与归档GPU一致。两个源窗口×两个容量×SRP/固定256/带保护256共12条Burst-stop轨迹，逐轮配额、空闲块、抢占和重算全部与GPU一致。',
        '- 重现旧SPARE归档轨迹哈希，并确认C的SPARE生成相同轨迹。缺失场景的两次确定性SRP模拟轨迹哈希一致；确定性重复不作为独立样本。',
        '- 使用原8个校准场景及全部24个测试场景。Fitted系数未重拟合，训练请求ID与C输入无交集。所有原始请求均值和配对汇总已独立复算。','',
        '## 与GPU逐场景可比的23场景','',
        '以下为逐场景TTFT改善百分比的均值；正数表示SPARE更快，负数表示SPARE更慢。不是总体均值之比。固定预算列始终比较原生256。','',
        '| 执行/时间模型 | 相对SRP | 相对原生固定256 | 相对带保护固定256 |',
        '| --- | ---: | ---: | ---: |']
    gp={r['reference']:r['descriptive_ttft_improvement_pct'] for r in gpu['paired_descriptive']}
    lines.append(f"| GPU（探索性23场景） | {gp['srp']:+.2f}% | {gp['fixed_selected']:+.2f}% | {gp['guarded_selected']:+.2f}% |")
    for p in ('gpu_fitted','resource','resource_staircase'):
        lines.append(f"| {names[p]}模拟 | {effect(p,'gpu_accepted23','srp'):+.2f}% | {effect(p,'gpu_accepted23','fixed_gpu256'):+.2f}% | {effect(p,'gpu_accepted23','guarded_gpu256'):+.2f}% |")
    lines+=['','Fitted与GPU在相对SRP和原生固定256的汇总方向接近，但不是逐场景完全一致；带保护固定256的增益方向反转，不能把模拟当成精确的硬件性能替代。','',
        '## 完整24场景及独立校准','',
        '| 时间模型 | 选定原生/保护预算 | 相对SRP | 相对校准原生固定 | 相对校准保护固定 | 相对GPU固定256 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for p in names:
        sel=summary['selection'][p]
        lines.append(f"| {names[p]} | {sel['fixed']}/{sel['guarded']} | {effect(p,'full24','srp'):+.2f}% | {effect(p,'full24','fixed_selected'):+.2f}% | {effect(p,'full24','guarded_selected'):+.2f}% | {effect(p,'full24','fixed_gpu256'):+.2f}% |")
    lines+=['','完整Fitted矩阵相对原生固定256为−6.74%（9胜/15负），相对保护固定256为−5.23%（5胜/19负）。GPU缺失场景的Fitted预测为SRP 2947.15 ms、SPARE 983.79 ms、原生固定256 482.30 ms：SPARE相对SRP很快，但仍慢于更小的固定预算。此预测不能填入GPU主表，也不能据此断言该场景真实GPU结果。','',
        '## A/A究竟发生了什么','',
        '三次失败尝试分别在GPU 6、1、1上发生，第二次重复均比第一次慢。首次分配分叉发生在第29/55/55轮，都是一侧已收到新请求、另一侧尚未收到：',
        '', '| 尝试 | 两次分叉步开始时间（ms） | 新请求到达（ms） |', '| --- | --- | --- |']
    aa=read(OUT/'gpu_aa_diagnosis.json')
    for r in aa['failed']:
        a,b=r['runs'];arrival=a['differing_requests'][0]['arrival_ms']
        lines.append(f"| {r['attempt']} | {a['divergent_start_ms']:.2f} / {b['divergent_start_ms']:.2f} | {arrival:.2f} |")
    lines+=['','三次总时长A/A相对差只有0.97%/1.87%/2.84%，TTFT相对差却为12.10%/9.41%/10.37%。调度轮数、抢占和重算也改变，不能按相同轮号把所有后续差异解释为同一批计算的计时噪声。',
        f"接受的31个校准/测试组中，{aa['second_slower']}组的第二次A/A TTFT更慢，中位增幅{aa['median_second_ttft_change_pct']:.2f}%。重复顺序/运行状态值得排查，但这些观察不能确定是GPU、CPU、温度还是其他因素造成。",'',
        '保持缺失场景输入与策略不变，只缩放Fitted的每步时间：','',
        '| 时间倍率 | SRP TTFT（ms） | SPARE TTFT（ms） | 原生固定256 TTFT（ms） |',
        '| --- | ---: | ---: | ---: |']
    indexed={(r['scale'],r['arm']):r for r in clock['rows']}
    for scale in (.98,.99,1.,1.01,1.02):
        lines.append(f"| {scale:.2f} | {indexed[scale,'srp']['ttft_ms']:.2f} | {indexed[scale,'spare']['ttft_ms']:.2f} | {indexed[scale,'fixed256']['ttft_ms']:.2f} |")
    lines+=['','SRP每步时间−1%/+1%时，平均TTFT为−7.09%/+3.79%；−2%/+2%时为−10.17%/+18.48%。说明在线到达与调度反馈足以放大小幅计时变化。这个确定性缩放诊断不是随机GPU噪声模型，也没有确定真实噪声来源。','',
        '## 设计和实现的判断','',
        '合理部分：固定与保护固定分别校准；测试请求不参与预算选择；执行相同排序及资源配置；正式方法顺序随机并在成对重复中反转；保留所有失败，没有为得到有利结果放宽门槛。',
        '',
        '需要改善的地方：',
        '',
        '1. 仅用两次SRP A/A和统一5%门槛判定整场景稳定，对时序敏感的在线负载较脆弱，也不能证明其他方法稳定。现有短预热仅4个256输入/16输出请求。下一版应先用独立pilot制定更充分的代表性预热、每臂重复和稳定性判断，记录GPU时钟/温度/功耗及CPU开销；不能看测试胜负再放宽门槛。',
        '2. c_policy.py第13–18行的保护固定控制先执行完整spare_select，再丢弃其选择、另算固定候选。这保持了决策语义，却包含固定控制不需要的候选计算，会增加其GPU实测中的CPU开销，不适合把相对保护固定的微小差异直接解释为省块门槛的纯收益。下一版需单独实现等价的基础计划/保护检查并测量开销。该问题偏向使保护固定变慢，不能解释为什么原生固定256仍快于SPARE。',
        '3. 只有两个校准源窗口、六个测试源窗口，容量也只覆盖768/2048。这是有限范围的独立窗口检验，不是任意负载下的最优预算证明。缺失场景不能当随机缺失处理。',
        '4. 模拟中不把各方法实际CPU调度耗时注入时钟；Fitted虽然包含历史engine.step平均开销，仍不能精确复现当前逐方法开销。Resource和Staircase未硬件校准。三模型用于检查结论敏感性，不构成跨硬件验证。','',
        '## 对当前Evaluation的意义','',
        '保持A的闭环主张，C仍只作探索性固定控制比较。现在有更多证据表明“固定256也能得到主要收益”必须认真对待；不能仅靠更换GPU或换成模拟结果来证明自适应增益。本次不改setting/method，不把模拟并入GPU表，也不替换缺失GPU数据。若继续推进方法创新，应在独立开发集定位选择性回退为何慢于固定256，再冻结新方法并重新做留出测试。','',
        '复现入口：`research/www_c_simulation_20261008/run.py`、`diagnose.py`；分析入口：`analyze.py`。模拟使用`.vllm-venv/bin/python -B`，分析可用`python3 -B`。协议见同目录PROTOCOL_CN.md及DIAGNOSTIC_PROTOCOL_CN.md。逐场景结果见paired.csv，校准及测试原始值见calibration.csv/test.csv，执行校验见checks.json。','']
    (OUT/'REPORT_CN.md').write_text('\n'.join(lines))
    (Path(__file__).parent/'RESULTS_CN.md').write_text('# C模拟结果\n\n已完成672条正式模拟和25条时序诊断，零失败。\n\n详见[完整报告](../../artifacts/www_c_simulation_20261008_r1/REPORT_CN.md)。\n')
    print('Verified 672 formal runs, 25 clock runs; wrote REPORT_CN.md and checks.json.')


if __name__=='__main__':main()
