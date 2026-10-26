"""Build reviewable descriptive evidence tables from completed, saved GPU runs."""
import collections
import csv
import math
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_saving_threshold_20260924.common import *
from research.v1_saving_threshold_20260924.analyze import load, group_rows


def expression(spec):
    if spec=='off':return '∞（不干预）'
    if isinstance(spec,(int,float)):return str(spec)
    if spec['kind']=='constant':return expression(spec['n'])
    if spec['kind']=='tree':
        t=spec['tree']
        return f"n(s) = {expression(t['left']['n'])}，若 {t['feature']} ≤ {t['cut']:g}；否则 {expression(t['right']['n'])}"
    terms=[f"{spec['intercept']:.6g}"]+[f"{c:+.6g} ln(1+{f})" for c,f in zip(spec['coefficients'],spec['features'])]
    z=' '.join(terms)
    return f'n(s) = clip(ceil(exp({z}) − 1), 1, 100000)'


def csv_write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    manifest,physical,rows=load()
    output=OUT/'analysis';output.mkdir(exist_ok=True,parents=True)
    summary=read(output/'summary.json')
    cells=[]
    for phase in ('map','refine'):
        for (case,cap),group in group_rows(rows,phase).items():
            if any(str(n) not in group for n in THRESHOLDS):continue
            tt={n:group[str(n)]['metrics']['mean_ttft_ms'] for n in THRESHOLDS}
            best=min(tt,key=tt.get)
            near=[str(n) for n in THRESHOLDS if tt[n]<=tt[best]*1.03]
            gain=100*(1-tt[best]/tt['off'])
            cells.append(dict(phase=phase,case=case,capacity=cap,best_threshold=str(best),
                near_best_within_3pct=' / '.join(near),best_ttft_ms=tt[best],off_ttft_ms=tt['off'],
                best_gain_percent=gain,n1_gain_percent=100*(1-tt[1]/tt['off']),
                n16_gain_percent=100*(1-tt[16]/tt['off']),
                category='observed_gain_ge_5pct' if gain>=5 else 'gain_below_5pct',
                best_preemptions=group[str(best)]['metrics']['preemptions'],
                off_preemptions=group['off']['metrics']['preemptions'],
                best_recompute=group[str(best)]['metrics']['recomputed_tokens'],
                off_recompute=group['off']['metrics']['recomputed_tokens']))
    cells.sort(key=lambda d:(d['case'],d['capacity']))
    csv_write(output/'capacity_cells.csv',cells)
    csv_write(output/'factor_pairs.csv',summary['factor_interactions'])
    formula=[]
    groups={}
    for r in rows:
        j=r['job']
        if j['phase']=='formula' and r['status']=='complete':groups.setdefault((j['case'],j['capacity']),{})[j['arm']]=r
    for (case,cap),g in sorted(groups.items()):
        if not all(a in g for a in ('formula','ablated','off')):continue
        m={a:r['metrics'] for a,r in g.items()}
        formula.append(dict(case=case,capacity=cap,formula_ttft_ms=m['formula']['mean_ttft_ms'],
            ablated_ttft_ms=m['ablated']['mean_ttft_ms'],off_ttft_ms=m['off']['mean_ttft_ms'],
            gain_over_off_percent=100*(1-m['formula']['mean_ttft_ms']/m['off']['mean_ttft_ms']),
            gain_over_ablated_percent=100*(1-m['formula']['mean_ttft_ms']/m['ablated']['mean_ttft_ms']),
            formula_preemptions=m['formula']['preemptions'],off_preemptions=m['off']['preemptions'],
            formula_recompute=m['formula']['recomputed_tokens'],off_recompute=m['off']['recomputed_tokens'],
            formula_e2e_ms=m['formula']['mean_e2e_ms'],off_e2e_ms=m['off']['mean_e2e_ms'],
            formula_p95_tbt_ms=m['formula']['p95_tbt_ms'],off_p95_tbt_ms=m['off']['p95_tbt_ms']))
    csv_write(output/'formula_validation.csv',formula)
    fit=read(OUT/'fit.json') if (OUT/'fit.json').exists() else None
    cpu=[read(p) for p in (OUT/'cpu/runs').glob('*.json')]
    cpucount=collections.Counter('complete' if r.get('complete') else r.get('status','incomplete') for r in cpu)
    cpu_rows=[]
    for r in sorted(cpu,key=lambda r:(r['case'],r['capacity'],str(r['threshold']))):
        cpu_rows.append(dict(case=r['case'],capacity=r['capacity'],threshold=r['threshold'],
            status='complete' if r.get('complete') else r.get('status','incomplete'),
            first_round_mean=st.fmean(r['first_round'].values()) if r.get('complete') else None,
            finish_round_mean=st.fmean(r['finish_round'].values()) if r.get('complete') else None,
            rounds=r.get('rounds'),preemptions=r.get('preemptions'),recompute=r.get('recomputed_tokens'),
            action_sha256=r.get('action_sha256')))
    csv_write(output/'cpu_replays.csv',cpu_rows)
    phases=collections.Counter(r['job']['phase'] for r in physical.values())
    audit=read(OUT/'audit.json') if (OUT/'audit.json').exists() else None
    result=dict(capacity_cells=cells,formula_validation=formula,cpu_counts=dict(cpucount),
        physical_phases=dict(phases),near_best_tolerance=.03,descriptive_gain_cut=.05,
        note='3% plateau and 5% useful labels are descriptive reporting bins, not statistical significance or new optimization constraints')
    write(output/'evidence.json',result)
    lines=['# 节约门槛研究：数据证据表','',
        '本文件由已保存结果生成；最终解释见研究目录中的 RESULTS_ANALYSIS_CN.md。',
        '门槛 n 表示候选至少节约多少物理 KV blocks 才值得采用。候选顺序、首 token/decoding 保护不变；没有额外压力开关。',
        '',f"物理执行：{len(physical)}；阶段计数：`{json.dumps(dict(phases),ensure_ascii=False)}`。非完成结果：{summary['failed']}。",
        f"CPU记录：`{json.dumps(dict(cpucount),ensure_ascii=False)}`。CPU轮数不换算GPU毫秒。",
        f"独立复算：{('通过，核对 '+str(audit['runs_checked'])+' 次运行') if audit and audit['passed'] else '待最终核验'}。",'',
        '## 1. 容量与门槛','',
        '每格单次运行。改善为相对同窗口、同容量 off 的 mean TTFT 降幅；正值更好。最好门槛是本次已测网格的描述性最好值。',
        '“近优门槛”指TTFT不超过本格最快值的1.03倍，用来显示平台；3%与下文5%仅为展示分档，不是显著性结论或新约束。',
        '', '| 输入 | KV blocks | 最好 n | 近优 n（3%） | 最好改善 | n=1改善 | n=16改善 |',
        '|---|---:|---|---|---:|---:|---:|']
    for d in cells:
        lines.append(f"| {d['case']} | {d['capacity']} | {d['best_threshold']} | {d['near_best_within_3pct']} | {d['best_gain_percent']:+.1f}% | {d['n1_gain_percent']:+.1f}% | {d['n16_gain_percent']:+.1f}% |")
    lines+=['','## 2. 受控因素','',
        'density：间隔从2倍原始变成1/2倍；burst：同跨度均匀到达变为4批；prompt：等总输入量的均匀长度变为原始长短组成；output：生成长度从约1/2倍变为2倍。',
        '“n=1相对n=16优势”直接比较两种门槛，正值表示n=1更快。这样可区分干预总体效果变化和门槛优劣变化。',
        '', '| 来源 | 因素 | KV | 低水平 n1/n16优势 | 高水平 n1/n16优势 | 低→高最好 n | n1收益交互(pp) | n16收益交互(pp) |',
        '|---|---|---:|---:|---:|---|---:|---:|']
    for d in summary['factor_interactions']:
        lines.append(f"| {d['source']} | {d['factor']} | {d['capacity']} | {d['low_n1_advantage_over_n16_percent']:+.1f}% | {d['high_n1_advantage_over_n16_percent']:+.1f}% | {d['low_best']}→{d['high_best']} | {d['interaction_n1_pp']:+.1f} | {d['interaction_n16_pp']:+.1f} |")
    if fit:
        lines+=['','## 3. 冻结公式与状态证据','',f"`{expression(fit['selected'])}`",'',
            f"选择来源：`{fit['origin']}`。容量变量 capacity 是总物理块数减1，扣除原生allocator保留块。",
            f"容量关系候选：`{expression(fit['capacity_formula'])}`。",
            f"按来源留出的容量关系/常数平均归一化损失：{fit['capacity_source_holdout_loss']:.6f} / {fit['constant_source_holdout_loss']:.6f}。",
            f"通过同状态核对的分支状态：{fit['branch_states']}；按前缀轨迹分组验证。分支标签是单轮动作改变后的整条轨迹TTFT差。",
            f"分支常数CV损失：{fit['branch_constant_cv']}。",'',
            '| 状态模型 | 输入 | 分组CV损失 |', '|---|---|---:|']
        for d in fit['state_candidates']:
            lines.append(f"| {d['family']} | {', '.join(d['features'])} | {d['cv_loss']:.6f} |")
        lines+=['',f"消融规则：`{expression(fit['ablation'])}`；被移除的输入：`{fit['ablated_feature']}`。",
            '公式使用决策前可观测状态；输出终长只用于受控负载构造，不是在线输入。']
    if formula:
        lines+=['','## 4. 新窗口公式检查','',
            '| 输入 | KV | 公式TTFT(ms) | off(ms) | 相对off改善 | 相对消融改善 |',
            '|---|---:|---:|---:|---:|---:|']
        for d in formula:
            lines.append(f"| {d['case']} | {d['capacity']} | {d['formula_ttft_ms']:.1f} | {d['off_ttft_ms']:.1f} | {d['gain_over_off_percent']:+.1f}% | {d['gain_over_ablated_percent']:+.1f}% |")
        plan=read(OUT/'plans/formula.json')
        lines+=['',f"输入不可行而预先跳过的规则组合：{len(plan.get('skipped',[]))}；精确相同规则复用：{len(plan.get('aliases',[]))}。"]
    lines+=['','## 5. 数学解释','',
        '精确新增量：G(a)=Σ max(0, ceil((computed_i+scheduled_i(a))/16)−held_i)。节约量为 d=G(base)−G(candidate)。',
        '固定状态的动作仅在门槛越过候选节约量时改变，因此允许使用门槛区间。',
        '在节约持续到资源冲突、单位缺口等待成本为r、当前延后等待代价有共同上界c的模型中：',
        '', '`n(s)=min{d≥1: r E[min(d,(Y−F)₊)|s]≥c}`；若没有可实现的d，则n=∞。',
        '', '这给出单一门槛的条件性收益保证，连接净需求Y、空闲F、恢复成本r和延后成本c；不将经验拟合称为全局最优证明。',
        '精确TTFT恒等式：Σ TTFT_i=∫U(t)dt，其中U(t)为已经到达但尚未得到首token的请求数。',
        '', '原始统计、E2E/TBT与重算见 runs.csv、capacity_cells.csv、factor_pairs.csv、formula_validation.csv。']
    (output/'EVIDENCE_CN.md').write_text('\n'.join(lines)+'\n')
    print('Evidence:',len(cells),'capacity cells;',len(formula),'formula pairs;',dict(cpucount),flush=True)


if __name__=='__main__':main()
