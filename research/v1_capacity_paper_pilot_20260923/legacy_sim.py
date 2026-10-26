"""Small supplemental replay of accepted-paper adapters in their existing engine.

No GPU latency claim: analytic costs and allocator/execution semantics differ.
"""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.v1_capacity_paper_pilot_20260923.common import *
from research.state_conditioned_prefill_selection_20260918.output_length_predictor import TrainOnlyLengthPredictor, _weighted_quantile
from run_paper_simulator_validation import run_case, IMPLEMENTATION_FILES

POLICIES = ('fcfs','orca-max','s3','ltr','mlfq','sarathi')
TARGET = OUT/'legacy_simulator_r2'


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    assert not (TARGET/'manifest.json').exists(), 'Never overwrite a frozen supplement'
    data = ROOT/'research/state_conditioned_prefill_selection_20260918/data'
    predictor = TrainOnlyLengthPredictor.from_data_dir(data)
    train_ids = set(predictor.request_rows)
    paths = {}
    for b in BLOCKS:
        reqs = read(INPUT/(b+'_original.json'))
        assert not train_ids & {r['id'] for r in reqs}
        records = []
        for i,r in enumerate(reqs):
            d = predictor.distribution(b.split('_')[0], r['prompt'])
            xs, ws = d['lengths'], d['weights']
            total = sum(ws)
            mean = sum(x*w for x,w in zip(xs,ws))/total
            std = math.sqrt(sum(w*(x-mean)**2 for x,w in zip(xs,ws))/total)
            records.append(dict(request_id=i, original_id=r['id'], arrival_ms=r['arrival_ms'],
                prompt_tokens=r['prompt'], actual_output_tokens=r['output'], max_new_tokens=2048,
                output_mean=mean, output_std=std,
                **{f'output_p{q}':_weighted_quantile(xs,ws,q/100) for q in (50,90,95,99)}))
        path = TARGET/'inputs'/(b+'.json')
        write(path,records); paths[b]=path
    old_defaults=read(ROOT/'configs/accepted_scheduler_baselines.json')['defaults']
    specs=[]
    for b in BLOCKS:
        for c in CAPS:
            for policy in POLICIES:
                simulator = dict(old_defaults['simulator'],token_budget_per_step=512 if policy=='sarathi' else 8192,
                    prefill_chunk_size=256,kv_capacity_tokens=(c-1)*16)
                specs.append(dict(old_defaults, name=f'{b}_k{c}_{policy}', policy=policy,
                    max_batch_size=32, trace=str(paths[b]), simulator=simulator,
                    write_details=True, max_steps=50000, wall_time_limit_s=30))
    source_files = list(IMPLEMENTATION_FILES)+[
        str((HERE/'legacy_sim.py').relative_to(ROOT)),
        'simulator_core/scheduler.py','simulator_core/kv_allocator.py',
        'research/state_conditioned_prefill_selection_20260918/output_length_predictor.py']
    manifest=dict(specs=specs, expected=36,
        claim='supplemental analytic simulator only; cannot compare milliseconds or wins with GPU/native-V1 tables',
        predictor='existing train-only prompt-neighbor distribution, not original S3/LTR model checkpoint',
        revision='Initial setup rejected before first episode: non-chunked prefill needs budget >= longest prompt. Use 8192 for full-prefill arms and archived Sarathi budget 512. No timing result existed before correction.',
        training={str(p.relative_to(ROOT)):digest(p) for p in sorted(data.glob('*_train.jsonl'))},
        inputs={b:digest(p) for b,p in paths.items()}, sources={p:digest(ROOT/p) for p in source_files})
    write(TARGET/'manifest.json',manifest)
    rows=[]
    for spec in specs:
        row=run_case(spec,ROOT,TARGET/spec['name']); rows.append(row)
        print(spec['name'],row['metrics']['completed'],row['metrics']['termination'],flush=True)
        write(TARGET/'status.json',dict(saved=len(rows),expected=36))
    write(TARGET/'summary.json',rows)
    lines=['# 已发表论文规则：补充模拟器回放','',
        '两源×三容量×六组=36段，同本轮64请求窗口及自然到达。没有调参、没有读取未来长度的oracle组。',
        'Orca/vLLM式FCFS、Orca-max、S³、LTR、MLFQ、Sarathi均复用原模拟器规则。',
        'S³/LTR使用已有训练集的prompt邻域预测；不是论文原预测模型。',
        '本表为未校准解析时间模型。不可与GPU毫秒、GPU排名直接合并，不用于宣称我们优于S³/LTR。',
        '容量按(c-1)×16可用tokens对齐，但执行/恢复语义仍不等于原生V1。',
        '完整prefill规则的budget=8192以容纳最长7686-token prompt；Sarathi沿用旧512。不是相同budget比较。',
        'S³的恢复为仓库丢弃重算而非原论文异步host swap；MLFQ是通用算法参照。','',
        '| 窗口/容量 | 方法 | 完成 | 模拟 mean TTFT (ms) | 模拟 mean E2E (ms) | 驱逐 |',
        '|---|---|---:|---:|---:|---:|']
    for spec,row in zip(specs,rows):
        m=row['metrics']; complete=m['completed']==m['requests']
        times=f"{m['ttft_ms_mean']:.1f} | {m['e2e_ms_mean']:.1f}" if complete else 'N/A | N/A'
        lines.append(f"| {spec['name'].rsplit('_',1)[0]} | {spec['policy']} | {m['completed']}/{m['requests']} | {times} | {m['evictions']} |")
    (TARGET/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    assert manifest['sources']=={p:digest(ROOT/p) for p in source_files}
    complete=sum(r['metrics']['completed']==r['metrics']['requests'] for r in rows)
    write(TARGET/'status.json',dict(finished=True,saved=len(rows),expected=36,complete=complete,
        failures=[r['name'] for r in rows if r['metrics']['completed']!=r['metrics']['requests']]))


if __name__=='__main__': main()
