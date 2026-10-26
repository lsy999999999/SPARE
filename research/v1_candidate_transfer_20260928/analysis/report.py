"""Readable interpretation of the completed, audited transfer experiment."""
import statistics as st,sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_candidate_transfer_20260928.common import *

def main():
    d=verify();summary=read(OUT/'analysis/summary.json');audit=read(OUT/'analysis/audit.json')
    exp=read(OUT/'analysis/explanation.json')['contrasts']
    bounds=read(OUT/'analysis/boundary_reference.json')['contrasts']
    status=read(OUT/'status.json');launch=read(OUT/'launcher.json')
    assert audit['passed'] and status['state']=='complete' and not summary['excluded']
    cs=summary['contrasts'];holes=[c for c in cs if c['kind']=='hole'];ks=[c for c in cs if c['kind']=='k_matched']
    assert len(cs)==24 and not summary['closed_loop_gate_passed']
    def find(group,kind):return next(c for c in cs if c['group']==group and c['kind']==kind)
    bad=find('code_fresh0_c1536_t1024_k128','hole')
    risk=find('conversation_fresh0_c768_t512_k256','hole')
    kp=find('code_fresh1_c1536_t512_k256','k_matched')
    small=find('conversation_fresh1_c768_t1024_k128','k_matched')
    risk_exp=next(c for c in exp if c['group']==risk['group'] and c['kind']=='hole')
    k_exp=next(c for c in exp if c['group']==kp['group'] and c['kind']=='k_matched')
    clock=lambda t:datetime.fromtimestamp(t,ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    lines=['# 二维候选新窗口复核：结果与解释','',
        '本轮进一步确认 K 不能省略，也确认“原四候选全失败就用 inverse(n) 补上”不够可靠。继续保留旧式＋A＋原四候选固定顺序；未部署新策略、未启动未通过门控的闭环实验。',
        '',f'完成 {summary["accepted_groups"]}/12 组、24 个同状态对照（12 门槛缺口、12 匹配 K）、58 个状态内去重动作。232 次正式分支、24 次 A/A、12 次 anchor，共 {audit["episodes"]} 次完整运行、{audit["rounds"]:,} 轮通过本地逐轮审计。无 GPU 污染排除。',
        f'GPU 流程 {clock(launch["started"])} 至 {clock(status["updated"])}，约 {(status["updated"]-launch["started"])/60:.2f} 分钟，包含远端 native 验证和模型启动，不是累计 GPU 分钟。',
        '', '数据来自四个新时间窗口，256 请求，与扫描的历史 inputs/raw_inputs 中 2624 个唯一 ID 无重叠；仍来自同一 Azure 周轨迹，不能称独立数据集。每来源两窗口，code/KV1536、conversation/KV768，各测基础 (1024,256)、(512,256)、(1024,128)，全部 burst_stop。',
        '', '所有状态、动作在分支性能数据前按状态规则确定，没有本轮发现阶段挑赢家。每动作四次交替顺序重复，只改变一轮，之后恢复 current。收益单位是分支时未首 token 请求的剩余等待面积/64，ms，不是完整新策略 mean TTFT 改善。',
        '', '## 1. 门槛缺口存在，但不足以决定选择',
        '',f'12 个缺口状态：稳定改善 {summary["hole"]["positive"]}，稳定退化 {summary["hole"]["negative"]}，其余 {12-summary["hole"]["positive"]-summary["hole"]["negative"]} 个未过复现门槛。闭环门控要求至少75%正向复现且没有反向复现等条件，本轮未通过。',
        '', '逆解使用当前状态、基础 (T,K)、G₀−n 和保护配额计算 T′，不是把 111、118 等实测结果固定成常数。四原候选失败包括省块不足或保护约束失败；合法缺口不等于收益缺口。',
        '',f'即使只差一个块也可能失败：code/fresh0、基础 (1024,128)、第 {bad["round"]} 轮，原候选最小省块缺口为 {bad["old_min_shortfall"]} 块，n={bad["n"]}。逆解 T′={bad["inverse_budget"]} 将实际 q 从 {bad["left_q"]} 降到 {bad["right_q"]}，新增块 {bad["left_growth"]}→{bad["right_growth"]}；当轮等待减少 {bad["first_gain_ms"]:.3f} ms，后续增加 {-bad["later_gain_ms"]:.3f} ms，净退化 {-bad["mean_gain_ms"]:.3f} ms。四次均过反向确认门槛。',
        '',f'更强的反例：conversation/fresh0、基础 (512,256)、第 {risk["round"]} 轮，q {risk["left_q"]}→{risk["right_q"]}，新增块 {risk["left_growth"]}→{risk["right_growth"]}，当轮多留下 {risk["left_growth"]-risk["right_growth"]} 块；当轮收益 {risk["first_gain_ms"]:.3f} ms，却有 {-risk["later_gain_ms"]:.3f} ms 的后续代价，净退化 {-risk["mean_gain_ms"]:.3f} ms。',
        f'该例四次的后续抢占分别是 {risk_exp["left_preemptions"]}→{risk_exp["right_preemptions"]}，重算分别是 {risk_exp["left_recomputed"]}→{risk_exp["right_recomputed"]}。减少本轮块增长并不保证减少后续抢占。请求推进、后续排序、驻留和释放路径必须一起看；这不是对每一项后续代价的独立因果分解。',
        '', '通过正向复现的缺口也完整保留：', '']
    for c in holes:
        if c['positive_confirmed']:
            lines.append(f'- {c["group"]} / 第{c["round"]}轮，inverse ({c["inverse_budget"]},{c["base_K"]})，q {c["left_q"]}→{c["right_q"]}，四次收益 {", ".join(f"{v:.3f}" for v in c["gains_ms"])} ms，均值 {c["mean_gain_ms"]:.3f} ms。')
    lines+=['', '## 2. K 在相同 q 和块增长下仍有作用','',
        f'12 个对照保持同一名义 T′、相同实际 q、相同新增块数，只让 K 改变逐请求分配。大 K 稳定较好 {summary["k_matched"]["large_better"]} 个，小 K 稳定较好 {summary["k_matched"]["small_better"]} 个，其余未过复现门槛。所有对照均保留在下表；此选点不代表线上出现频率，也不证明大 K 在所有状态都更好。',
        '',f'例如 code/fresh1、基础 (512,256)、第 {kp["round"]} 轮，两边 q={kp["left_q"]}、新增 {kp["left_growth"]} 块、125 个 prefill token，半成品数均为2。K256 比 K64 四次分别少等 {", ".join(f"{-v:.3f}" for v in kp["gains_ms"])} ms，均值 {-kp["mean_gain_ms"]:.3f} ms。当轮略慢 {kp["first_gain_ms"]:.3f} ms，后续少等 {-kp["later_gain_ms"]:.3f} ms。',
        '该例大 K 将125 token全部推进剩余574 token的请求；小 K 分成64和61，把61给剩余2423 token的另一请求。总推进、总块增长、半成品数都相同，完成距离的分布不同。这支持评价“推进了谁”，不能只用 q、growth 或半成品数量作为动作价值。',
        '',f'反方向也复现：conversation/fresh1、基础 (1024,128)、第 {small["round"]} 轮，同 q={small["left_q"]}、同新增 {small["left_growth"]} 块，K32 比 K128 四次分别少等 {", ".join(f"{v:.3f}" for v in small["gains_ms"])} ms，均值 {small["mean_gain_ms"]:.3f} ms。',
        '该状态有一个已输出61个token、正在恢复重算的请求。小 K 将它的本轮配额从128压到32，把未首token请求的总推进从128提高到224，同时五个正常decode各1token不变；并把推进分散给更多未首token请求。半成品数1→7，后续抢占124→132，重算83262→82867，等待反而改善。故不能统一规定大K更好、半成品越少越好或抢占越少越好。这识别了首token推进与已首token恢复重算之间的分配差异，但还没有把各项分配改变的贡献分别隔离。',
        '', 'K 对照的大者优于小者，不自动表示它优于 current。第三参照的全部比较另存 explanation.json 的 vs_current，标为补充描述性分析，不从中再挑策略赢家。',
        '', '## 3. 同状态补充证据支持 256 右侧的硬件台阶','',
        '从已经测完的参照中，按动作结构筛出 current q 为128/256/512、另一动作只多1–2token、K与块增长相同、各请求推进不减少的比较。筛选不看测得收益，但这是补充分析，不是预注册主终点。',
        '', '|组/轮|q|K|块增长|非调度耗时增加（均值比）|', '|---|---|---:|---:|---:|']
    for b in bounds:
        lines.append(f'|{b["group"]}/{b["round"]}|{b["q"]}|{b["chunk"]}|{b["growth"]}|{100*(b["mean_ratio"]-1):.2f}%|')
    lines+=['', '这些比较进一步支持原 T1024 配置下 q256 的右侧耗时台阶；非调度耗时包含非调度 host 开销，不能据此指定某个 kernel 原因。多推进1–2token后的净等待并不都稳定改善或退化。它们也不能证明 T/4 的比例对任意基础 T 都最优，更没有独立分摊比例结构和硬件效率的贡献。',
        '', '## 4. 全部预设对照','',
        '正值：缺口对照表示 inverse 改善；K 对照表示小 K 改善。确认要求四次差值都同号且每次绝对值超过 max(1ms, 两动作各自四次极差的最大值)，不是统计显著性检验。',
        '', '|组|类型/轮|四次收益 ms|均值 ms|确认|', '|---|---|---|---:|---|']
    for c in cs:
        direction='正向' if c['positive_confirmed'] else '反向' if c['negative_confirmed'] else '未过'
        values=' / '.join(f'{v:+.3f}' for v in c['gains_ms'])
        lines.append(f'|{c["group"]}|{c["kind"]}/{c["round"]}|{values}|{c["mean_gain_ms"]:+.3f}|{direction}|')
    lines+=['', '## 5. 波动、别名与解释边界','',
        '没有删除大波动重复或按结果补跑。例如code/fresh0/T1024/K128、第253轮的一个小K动作，四次物理轨迹完全相同，其中第四次后续第357轮step约344ms，其余参考约30ms，而schedule约0.48ms。这是非调度部分的运行时抖动证据，具体原因未确认；仍保留该重复，该K对照按预设规则未确认。完整重复轨迹摘要见 variability.json。',
        '', '实际 current 参数与去重代表别名可能不同。例如某状态的 T′=512、768、1024 产生完全相同的计划。主表的 q/growth 来自真实执行，actual_current 保存anchor实际参数，aliases保存等价参数；不能把别名差异当作预算收益。',
        '', '本轮给出的更具体解释是：硬件台阶影响本轮执行成本；K影响对哪些请求集中推进；候选池的离散档位同时限制哪些干预能发生。原池的T方向最低档T/4、K方向最低档min(64,K)不是实际q的严格下限，更不是已经证明最优的比例。',
        '', '下一步优先把未首token推进与已首token请求的恢复重算分开，在相同q/growth下检验恢复配额和完成距离的贡献，再考虑KV占用归属和释放路径。不再把“满足n”“多省块”或“q/耗时大”单独当作选择依据。先用这些完整正负对照检验一个可解释条件，再在另外的新窗口做闭环。本轮没有得到足以替换原四候选的通用规则。',
        '', '复现入口：prepare.py → launch.py；本地analyze.py逐轮重建全部记录；analysis/explain.py、check_variability.py、check_boundary.py、report.py生成补充解释。冻结设计/源码哈希在 artifacts/v1_candidate_transfer_20260928_r1/design.json，启动前import修正记录在prelaunch_amendment.json。']
    (HERE/'ANALYSIS_CN.md').write_text('\n'.join(lines)+'\n')
    print('Wrote',HERE/'ANALYSIS_CN.md')

if __name__=='__main__':main()
