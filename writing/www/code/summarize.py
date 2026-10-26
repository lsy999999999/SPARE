"""Summarize separate replay tiers and diagnose existing matched GPU windows."""
import csv
import json
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from run_paper_simulator_validation import error_metrics
from writing.www.code.run_simulator import (read, write, csv_write, digest,
                                            DEFAULT_OUTPUT, ARMS)


def main():
    results = read(DEFAULT_OUTPUT/'summary.json')
    expected = 32 * len(ARMS)
    assert len(results) == expected and all(
        r['complete'] and r['metrics']['completed'] == 64 for r in results)
    by = {r['name']:r for r in results}
    lines = ['# 本轮 simulator baseline 对比结果（2026-09-29）', '',
        f'原生 V1 CPU 完成{expected}/{expected}条轨迹（32场景×{len(ARMS)}方法），每条64请求。主方法为 arrival；GPU调用数为0。时间为未校准合成假设，以下不能作为已验证的真实硬件收益。名称含 oracle 的方法读取真实输出长度且不可部署；rule/proxy 是论文规则在共享 KV-only 后端上的适配。', '',
        '## 同场景相对当前 arrival 的 mean TTFT 改善', '',
        '先在每个场景计算 (arrival−方法)/arrival，再按场景等权平均；负值表示比当前方法更差。不同时间模型分开，宽松容量也保留。', '',
        '|时间模型|方法|场景数|平均改善|最差改善|总抢占|总重算token|',
        '|---|---|---:|---:|---:|---:|---:|']
    for profile in ('resource','resource_staircase'):
        for arm in ARMS:
            rows=[r for r in results if r['arm']==arm and r['profile']==profile]
            gains=[]
            for r in rows:
                prefix=r['name'][:-len(arm)]
                base=by[prefix+'arrival']['metrics']['ttft_ms_mean']
                gains.append(100*(base-r['metrics']['ttft_ms_mean'])/base)
            lines.append(f"|{profile}|{arm}|{len(rows)}|{st.fmean(gains):+.2f}%|{min(gains):+.2f}%|{sum(r['metrics']['preemptions'] for r in rows)}|{sum(r['metrics']['recomputed_tokens'] for r in rows)}|")
    lines += ['', '## 32场景等权的绝对指标', '',
        '每个单元格先在单场景64请求内聚合，再对32个场景等权平均。P95/P99因此是“场景内分位数的平均”，不是混合2048请求后的分位数。', '',
        '|方法|TTFT mean|TTFT P95|TTFT P99|E2E mean|E2E P95|E2E P99|TPOT mean|max ITL P99|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rows = [r['metrics'] for r in results if r['arm'] == arm]
        mean = lambda field: st.fmean(r[field] for r in rows)
        lines.append(f"|{arm}|{mean('ttft_ms_mean'):.1f}|{mean('ttft_ms_p95'):.1f}|{mean('ttft_ms_p99'):.1f}|{mean('e2e_ms_mean'):.1f}|{mean('e2e_ms_p95'):.1f}|{mean('e2e_ms_p99'):.1f}|{mean('tpot_ms_mean'):.1f}|{mean('max_itl_ms_p99'):.1f}|")
    lines += ['', '|方法|request/s|output token/s|SLO达标率|SLO goodput|peak KV|时均KV|总抢占|总重算token|调度CPU mean|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rows = [r['metrics'] for r in results if r['arm'] == arm]
        mean = lambda field: st.fmean(r[field] for r in rows)
        lines.append(f"|{arm}|{mean('request_throughput_rps'):.3f}|{mean('output_throughput_tps'):.1f}|{100*mean('slo_ttft5000_maxitl200_attainment'):.1f}%|{mean('slo_ttft5000_maxitl200_goodput_rps'):.3f}|{100*mean('peak_kv_util'):.1f}%|{100*mean('time_weighted_kv_util'):.1f}%|{sum(r['preemptions'] for r in rows)}|{sum(r['recomputed_tokens'] for r in rows)}|{mean('cpu_schedule_including_audit_ms_mean'):.3f} ms|")
    lines += ['', '完整指标为 [metrics.csv](native_simulator_20260929/metrics.csv)，每个子目录含 result.json、requests.csv、iterations.jsonl.gz。控制器开销列包含审计成本，不是GPU kernel时间。', '',
        '## 用既有 GPU 记录检查当前时间假设', '',
        '以下使用已有比例实验 fixed64 臂（与 arrival 同规则）的8个匹配场景、每场景2次GPU重复。核对请求ID、长度、入口到达、T/K和容量，按请求配对。该诊断未作拟合；历史窗口已用于研究探索，不能当作独立留出验证。', '',
        '|合成时间模型|GPU重复×场景|配对请求|TTFT MAE(ms)|TTFT MAPE|TTFT bias(ms)|E2E MAPE|',
        '|---|---:|---:|---:|---:|---:|---:|']
    comparisons=[]
    gpu_root=ROOT/'artifacts/v1_ratio_closed_loop_20260928_r1'
    for r in results:
        if r['arm']!='arrival' or r['capacity_configured_blocks']==4096:
            continue
        input_name=Path(r['input']).stem
        shape='burst_stop' if input_name.endswith('burst_stop') else 'steady'
        source_window=input_name[:-(len(shape)+1)]
        group=f"{source_window}_c{r['capacity_configured_blocks']}_t{r['budget']}_k{r['chunk']}"
        sim_rows={row['request_id']:row for row in csv.DictReader((DEFAULT_OUTPUT/r['name']/'requests.csv').open())}
        inputs={x['id']:x for x in read(ROOT/r['input'])}
        assert digest(gpu_root/'inputs'/Path(r['input']).name)==r['input_sha256']
        for repeat in (0,1):
            path=gpu_root/'groups'/group/'runs'/f'{shape}_fixed64_r{repeat}.json'
            gpu=read(path)
            assert gpu['status']=='complete' and set(gpu['clocks'])==set(sim_rows)
            assert {x['id']:x for x in gpu['requests']} == inputs
            assert gpu['config']['budget']==r['budget'] and gpu['config']['chunk']==r['chunk']
            assert gpu['config']['concurrency']==r['concurrency']
            assert gpu['config']['threshold']['tau_s']==1
            assert gpu['config']['threshold']['candidate_mode']=='fixed64'
            for rid,clocks in gpu['clocks'].items():
                sim=sim_rows[rid]
                assert clocks['tokens']==inputs[rid]['output']
                assert abs(clocks['arrival_ms']-inputs[rid]['arrival_ms'])<1e-7
                comparisons.append(dict(profile=r['profile'],case=r['name'],repeat=repeat,request_id=rid,
                    gpu_source=str(path.relative_to(ROOT)),gpu_sha256=digest(path),
                    sim_ttft_ms=float(sim['ttft_ms']),gpu_ttft_ms=clocks['first_ms']-clocks['arrival_ms'],
                    sim_e2e_ms=float(sim['e2e_ms']),gpu_e2e_ms=clocks['finish_ms']-clocks['arrival_ms']))
    aggregate={}
    for profile in ('resource','resource_staircase'):
        rows=[r for r in comparisons if r['profile']==profile]
        aggregate[profile]={key:error_metrics([r[f'sim_{key}_ms'] for r in rows],[r[f'gpu_{key}_ms'] for r in rows]) for key in ('ttft','e2e')}
        t,e=aggregate[profile]['ttft'],aggregate[profile]['e2e']
        lines.append(f"|{profile}|{len(set((r['case'],r['repeat']) for r in rows))}|{len(rows)}|{t['mae_ms']:.2f}|{t['mape_pct']:.2f}%|{t['bias_ms']:.2f}|{e['mape_pct']:.2f}%|")
    write(DEFAULT_OUTPUT.parent/'gpu_timing_diagnostic.json',dict(independent_holdout=False,performance_validated=False,
        source='existing fixed64 GPU runs; no new GPU execution',metrics=aggregate))
    csv_write(DEFAULT_OUTPUT.parent/'gpu_request_errors.csv',comparisons)
    destinations = {
        'ssjf_oracle': ('proxy_ssjf_2024', 'current_native_oracle_summary.json'),
        'mc_sf_oracle': ('online_kv_2025', 'current_native_oracle_summary.json'),
        'trail_oracle': ('trail_2025', 'current_native_oracle_summary.json'),
        'tie_oracle_proxy': ('tie_2026', 'current_native_oracle_proxy_summary.json'),
        'uniboost_static': ('uniboost_2026', 'current_native_rule_summary.json'),
        'apt_serve_kv_rule': ('apt_serve_2025', 'current_native_rule_summary.json'),
        'sola_rule': ('sola_2025', 'current_native_rule_summary.json'),
        'sarathi': ('sarathi_2024', 'current_native_summary.json'),
        'past_future': ('past_future_2025', 'current_native_summary.json'),
    }
    for arm, (directory, filename) in destinations.items():
        arm_rows = [r for r in results if r['arm'] == arm]
        write(ROOT/'writing/www/baselines'/directory/'results'/filename, dict(
            arm=arm, performance_validated=False,
            deployable=all(r['deployable'] for r in arm_rows),
            adaptation_level=arm_rows[0]['adaptation_level'],
            paper_mechanism_omitted=arm_rows[0]['paper_mechanism_omitted'],
            warning='Synthetic timing on the shared native V1 KV-only backend.',
            results=arm_rows))
    lines += ['', '**当前不能写“simulator的GPU性能预测已验证正确”。** 原生分配和动作一致性已经有测试，时间精度仍需要独立profiling与留出验证。不要用这张诊断表反过来选择有利测试窗口。', '',
        '## 历史论文规则重跑', '',
        '通用引擎的22项实验另见 [报告](legacy_rules_replay_20260928/report.md)。包含FCFS、Orca Max/Pow2/Oracle、S³代理、LTR代理/Oracle、MLFQ、Sarathi、Past-Future规则和旧AT-BB。它们不与本页原生V1结果交叉排名。', '',
        '原文与作者代码、规则改动、预测器缺口见 [baseline审计](../BASELINE_AUDIT_CN.md)。TRAIL/TIE 使用真实长度构造不可部署 oracle/proxy；UniBoost 固定 gamma；Apt-Serve 去掉 hidden cache；SOLA 使用当前合成 cost model，具体差异写入逐篇 ADAPTATION_CN.md。']
    (DEFAULT_OUTPUT.parent/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    main()
