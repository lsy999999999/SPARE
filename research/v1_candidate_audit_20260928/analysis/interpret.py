"""Post-run interpretation; keep the frozen runtime and its hashes unchanged."""
import csv
import statistics as st
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_candidate_audit_20260928.common import *
from research.v1_candidate_audit_20260928.analyze import load_rounds, csv_write
from research.v1_candidate_audit_20260928.run import verify,select_targets


def main():
    d=verify();a=read(OUT/'analysis/audit.json');assert a['passed'] and not a['excluded']
    assert read(OUT/'validation.json')['source_hashes']==d['source_hashes']
    states=read(OUT/'analysis/summary.json')['states'];phases=Counter();details=[];pool_plans=0
    for group in d['groups']:
        p=OUT/'groups'/group['key'];anchor=load_rounds(read(p/'runs/anchor.json'))
        selected=read(p/'selected.json')
        assert selected==select_targets(anchor,group['budget'],group['chunk'])
        for runpath in (p/'runs').glob('*.json'):phases[read(runpath)['phase']]+=1
        for target in selected['states']:
            pool_plans+=len(target['actions']);index=target['round']
            state=next(s for s in states if s['group']==group['key'] and s['target']==index)
            winners=read(p/f's{index}_winners.json')['winners']
            chosen={role:next(x for x in target['actions'] if x['id']==aid) for role,aid in winners.items()}
            labels={role:[read(p/f'runs/s{index}_confirm_{aid}_r{r}.json')['label'] for r in range(2)] for role,aid in winners.items()}
            # Dedupe representatives need not equal the actual current T/K.
            state['current_actual']=f"({anchor[index]['budget']},{anchor[index]['chunk']})"
            state['current_representative']=state['current']
            state['winner_is_base']='base' in chosen['best_expanded']['families']
            means={role:{k:st.fmean(x[k] for x in vals) for k in
                ('area_per_request_ms','first_round_ms','later_ms','step_ms','q','growth','preemptions','recomputed_tokens','kv_block_ms_approx')}
                for role,vals in labels.items()}
            details.append(dict(group=group['key'],target=index,threshold=target['threshold'],free=target['free'],
                current_actual=state['current_actual'],current_representative=state['current_representative'],
                chosen=chosen,confirmation_means=means,
                first_round_gain=means['current']['first_round_ms']-means['best_expanded']['first_round_ms'],
                later_gain=means['current']['later_ms']-means['best_expanded']['later_ms'],
                old_proposals=[x for x in target['proposals'] if 'old' in x['families']]))
    assert pool_plans==88 and sum(phases.values())==a['episodes']==230
    expanded=[s for s in states if s['confirm_expansion_gain']>0 and s['directional_confirmed']]
    retained=[s for s in states if s['confirm_total_gain']==0]
    basewins=[s for s in states if s['winner_is_base'] and s['directional_confirmed']]
    assert len(expanded)==3 and len(basewins)==1 and len(retained)==6
    write(OUT/'analysis/interpretation.json',dict(script_sha=digest(Path(__file__)),
        selection_reconstruction_passed=True,phases=dict(phases),distinct_state_actions=pool_plans,
        confirmed_expansion_states=len(expanded),confirmed_base_states=len(basewins),unchanged_winner_states=len(retained),
        states=states,details=details))
    contrasts=list(csv.DictReader((OUT/'analysis/k_contrasts.csv').open()))
    matched=[r for r in contrasts if r['equal_q']==r['equal_growth']=='True']
    pair=next(r for r in matched if r['group']=='conversation_c768_t1024_k256' and r['target']=='406' and r['a']=='a00' and r['b']=='a05')
    cv=read(OUT/'coverage.json')['runs'];coverage={k:sum(r.get(k,0) for r in cv) for k in
        ('rounds','normal_rounds','k_effective','k_matched','with_new_plans','new_plans')}
    start=read(OUT/'launcher.json')['started'];end=read(OUT/'status.json')['updated']
    lines=['# 二维候选机会审计：结果与解释','',
        '本轮发现原四候选之外存在可重复的单动作收益，K也确实影响分配与等待。建议继续研究按状态生成候选，同时保留current作为主方案；尚未测得一个可以替换current的闭环策略。',
        '',f'六组全部完成、无污染排除；{a["episodes"]}次运行、{a["rounds"]:,}轮通过本地独立重建。12个状态、88个不同合法动作；176次发现分支、36次独立确认、12次A/A、6次锚点。GPU控制流程约{(end-start)/60:.2f}分钟（含远端native验证与模型启动，不是累计GPU分钟）。',
        '', '负载：conversation/KV768、code/KV1536，各自基础(T,K)=(1024,256),(512,256),(1024,128)。均复用w0、64请求全部起点到达。状态在看标签前冻结，K有效/无效状态各优先一个；该选点比例不是线上频率。',
        '', '## 1. 候选覆盖与实验单位','',
        f'旧日志20条轨迹{coverage["rounds"]:,}轮，正常计划{coverage["normal_rounds"]:,}轮；其中{coverage["k_effective"]:,}轮存在K改变实际合法token分配，{coverage["k_matched"]:,}轮存在同名义T、同实际q、同块增长的不同K计划。{coverage["with_new_plans"]:,}轮有原池以外的新计划。{coverage["new_plans"]:,}是“状态×计划”计数，不是独立性能样本。',
        '', '扩展池包括比例网格、绝对64档位、物理省块目标反算和最多两个完成边界。通过n、保护、物理检查后，按逐请求token分配去重，所有不同合法计划均测两遍。原池在本报告中指原四候选加基础计划；基础计划是“不干预”对照，不能把选基础计划的收益称作仅调整四候选排序。',
        '', '## 2. 三个新增候选的独立确认','',
        '下表收益为同状态单轮替换后，残余首token等待面积/64的减少，单位ms；不是完整新策略的百分比收益。',
        '', '|负载 / KV|基础(T,K)|轮|新增动作(T′,K′)|实际q|两次改善|平均改善|',
        '|---|---|---:|---|---:|---|---:|']
    for s in expanded:
        detail=next(x for x in details if x['group']==s['group'] and x['target']==s['target'])
        q=detail['chosen']['best_expanded']['q']
        lines.append(f"|{s['group'].split('_t')[0]}|({s['base_T']},{s['base_K']})|{s['target']}|{s['best_expanded']}|{q}|{s['total_r0']:.3f} / {s['total_r1']:.3f}|{s['confirm_total_gain']:.3f}|")
    lines+=['', '三个动作均来自事先定义的inverse生成家族；111、118、16都是这些状态的计算结果，不能直接固定成新的常数表。原始(1024,256)配置也有一个新增候选获益，另外两个发生在基础T或K改变后。',
        '', '## 3. 可以具体解释的候选缺口','',
        'code/T512/K256，第32轮：基础计划增长32块，n=25。原T/4=128计划增长8块，只省24块，差1块未达门槛；其他原候选也未达标，因此current执行基础计划。逆解得到(T′,K′)=(118,256)，增长7块，恰好省25块，补上原比例候选遗漏的合法动作。两边都无抢占重算；当前轮约50.6ms降至23.5ms，本轮等价等待减少约18.67ms，后续等待增加约7.86ms，净改善10.81ms。',
        '', 'conversation/T1024/K256，第316轮：基础名义预算1024，实际只执行261token、增长17块；n=10。T/4动作无法省出10块，较大预算的某些参数与基础计划等价。逆解(111,256)增长7块，满足省10块；确认改善2.92ms，其中本轮约2.49ms、后续约0.43ms。抢占仍为84次，重算少150token；不能把全部收益归因于避免抢占。',
        '', 'conversation/T1024/K128，第342轮：原(256,128)实际q256、增长16块、半成品2个；新(1024,16)实际q213、增长13块、半成品13个。新动作保持总预算上限但由小K限制执行形状。本轮等待略增，后续等待减少，确认净改善12.90ms；抢占166→175、重算96887→96353。更少半成品、更少抢占均不是此例的收益解释。需要后续路径对照才能进一步分离空间、请求分配与释放的因果贡献。',
        '', '## 4. K的作用与无效参数变化','',
        f'共记录{len(matched)}个同名义T、同实际q、同块增长的不同K动作对。这些是发现阶段的完整对照，不是新增的独立确认样本。',
        '', f'例如conversation/T1024/K256的第406轮，(256,256)与(256,64)均执行256token、增长相同；K256比K64两次分别少等待{-float(pair["r0"]):.3f}、{-float(pair["r1"]):.3f}ms。说明只用q和growth不能表示整个动作价值。这一对照未另行追加独立确认，不把它报告成确认阶段赢家结果。',
        '', '另外，部分名义K变化完全不改变实际计划，这些已去重。小K成功例与大K较优对照共同说明：K不能删除，也不能规定越小或越大越好。',
        '', '## 5. 全部状态的确认结果','',
        '|组|轮|current实际参数|扩展池赢家代表参数|扩池收益ms|旧池选择/不干预收益ms|总收益ms|方向确认|',
        '|---|---:|---|---|---:|---:|---:|---|']
    for s in states:
        lines.append(f"|{s['group']}|{s['target']}|{s['current_actual']}|{s['best_expanded']}|{s['confirm_expansion_gain']:+.3f}|{s['confirm_selection_gain']:+.3f}|{s['confirm_total_gain']:+.3f}|{'通过' if s['directional_confirmed'] else '未通过/同动作'}|")
    lines+=['', '方向确认是预先定义的描述性规则：两次收益均超过max(1ms,确认动作各自重复极差的最大值)，不等于统计显著性或总体置信界。',
        '', '12状态中，3个新增候选通过方向确认；1个基础计划通过（code/T1024/K256第124轮，+7.63ms）；2个基础计划平均收益不足1ms；6个状态赢家仍为current。没有把不确定或无收益的状态删除。',
        '', '参数别名说明：自动states.csv中的current列是去重后的代表参数。conversation/T1024/K256第316轮的(512,256)、(768,256)、(1024,256)同计划，自动表用(512,256)代表；锚点current实际选择(1024,256)。上表已明确使用锚点实际参数。该表示差别不改变测得的q、growth及等待收益，全部别名保存在selected.json。',
        '', '## 6. 下一步判断','',
        '应该继续研究候选生成，优先验证“原比例候选刚好跨不过n的物理块边界”这一具体缺口，并保留K产生的不同推进分布。同一n下可由G0−g反算T′，输出随当前(T,K)、请求已占块/块内空余、保护配额变化；硬件时间曲线用于评价这些实际计划，不直接把111/118或16硬编码。三个确认赢家来自已尝试过的inverse家族，所以本轮也支持此前动态策略失败不能单独否定候选生成家族；仍需解决选择问题。',
        '', '下一轮先在独立窗口收集这类“旧池无达标节约、新池有达标计划”的状态，同时对K有效状态保留同q/同增长对照，检验候选生成与价值预测的可迁移性；有稳定选择条件后再验证低开销闭环。当前证据不能批准全局改用更小T、更小K或把所有inverse候选优先。',
        '', '本轮只分别改变基础T与K，并在每组内部做同状态分支；跨基础配置的锚点不同。因此尚未完整回答比例结构与硬件绝对点各占多大贡献。复用两个来源窗口、只测all-arrived/current锚点、每状态有限候选和两次确认，也限制了总体/steady到达结论。',
        '', '审计文件：analysis/audit.json、analysis/interpretation.json；全候选标签analysis/actions.csv，K对照analysis/k_contrasts.csv。冻结协议PROTOCOL_CN.md，后处理入口analysis/interpret.py。']
    (HERE/'ANALYSIS_CN.md').write_text('\n'.join(lines)+'\n')
    (OUT/'ANALYSIS_CN.md').write_text('\n'.join(lines)+'\n')
    print(dict(selection_reconstruction_passed=True,phases=dict(phases),confirmed_expansion_states=len(expanded),matched_k_pairs=len(matched)))


if __name__=='__main__':main()
