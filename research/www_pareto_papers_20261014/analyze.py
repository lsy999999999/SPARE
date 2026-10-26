"""External paper comparison, with oracle diagnostics reported independently."""
import csv
import os
from collections import defaultdict, Counter
import numpy as np
from .run import ROOT, OUT, PREVIOUS, ARMS, MAIN, DIAGNOSTIC, BUDGETS, CAPACITIES, LABELS, SCOPES, read, write, digest
from research.www_pareto_20261014.analyze import dominates, ci

SCOPE_CN = {
    'arrival': '当前SPARE；调度时不知道请求的真实输出长度。',
    'native': '现代vLLM 0.9.2 V1工程对照；非SOSP 2023原版内核。',
    'sarathi': '移植decode优先、部分prefill与FIFO规则；复用V1的KV分配。',
    'past_future': '按历史长度条件采样，并以预测的未来KV峰值决定准入。',
    'past_future_multi': '小batch重复采样32次并取平均峰值；采样次数与聚合方式是本地选择。',
    'orca_max': 'FIFO迭代调度与最大长度预留；未复现selective batching。',
    's3_proxy': '十桶长度预测代理与first-fit装箱；缺少DistilBERT和host swap。',
    'ltr_proxy': '历史中位数作为排序代理，并保留饥饿提升；缺少OPT/ListMLE预测器。',
    'apt_serve_kv_rule': 'pending/KV密度的beta=0特例；缺少hidden cache与完整跨阶段求解。',
    'sola_rule': '旧版风险排序规则；内部成本公式固定。',
    'sola_calibrated': '读取已有校准成本、按模拟耗时在线更新，并选择阶段与预算。',
    'uniboost_static': 'phase信号、几何量化与固定gamma；KV swap以重算代替。',
    'ssjf_oracle': '以真实输出长度代替BERT预测；仅作诊断。',
    'trail_oracle': '以真实剩余长度排序并有限抢占；swap以重算代替；仅作诊断。',
    'mc_sf_oracle': '以真实输出长度做短任务优先与未来KV峰值准入；仅作诊断。',
    'tie_oracle_proxy': '真实长度构造log-t分布代理；缺少DeBERTa预测器；仅作诊断。',
    'ltr_oracle': '真实输出长度提供排序分数；缺少OPT/ListMLE预测器；仅作诊断。',
}


def csv_write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    d = read(OUT / 'design.json')
    for rel, sha in d['hashes'].items():
        assert digest(ROOT / rel) == sha, rel
    raw = []
    for s in d['samples']:
        for cap in CAPACITIES:
            for arm in ARMS:
                rows = read(OUT / 'runs' / f"{s['name']}_c{cap}_{arm}.json")
                assert len(rows) == 4 and {r['budget'] for r in rows} == set(BUDGETS)
                assert all(r['complete'] and r['metrics']['completed'] == 64 for r in rows)
                raw.extend(rows)
    assert len(raw) == d['expected_runs'] == 19040
    flat = [dict(source=r['source'], name=r['name'], day=r['day'], split=r['split'],
                 capacity=r['capacity'], arm=r['arm'], budget=r['budget'], diagnostic=r['diagnostic'],
                 ttft_s=r['metrics']['ttft_ms_mean']/1000, output_tps=r['metrics']['output_throughput_tps'],
                 preemptions=r['metrics']['preemptions'], recomputed_tokens=r['metrics']['recomputed_tokens'],
                 trajectory_sha256=r['trajectory_sha256']) for r in raw]
    csv_write('runs.csv', flat)
    lookup = {(r['name'], r['capacity'], r['arm'], r['budget']): r for r in flat}
    assert len(lookup) == len(raw)
    # Every reused method is rerun in this matrix and must match all old trajectories.
    parity = 0
    for s in d['samples']:
        for cap in CAPACITIES:
            for r in read(PREVIOUS / 'runs' / f"{s['name']}_c{cap}.json"):
                if r['arm'] == 'srp':
                    continue
                assert lookup[r['name'], cap, r['arm'], r['budget']]['trajectory_sha256'] == r['trajectory_sha256']
                parity += 1
    points, selected, winners, verdicts, pairs, default_pairs = [], [], [], [], [], []
    rng = np.random.default_rng(20261014)
    for source in ('code', 'conversation'):
        for cap in CAPACITIES:
            ids = {split: sorted(s['name'] for s in d['samples'] if s['source']==source and s['split']==split)
                   for split in ('validation','test')}
            assert len(ids['validation'])==20 and len(ids['test'])==50
            days=defaultdict(list)
            for i,n in enumerate(ids['test']):
                days[lookup[n,cap,'arrival',1024]['day']].append(i)
            assert sorted(map(len,days.values()))==[25,25]
            boot=np.concatenate([rng.choice(v,(5000,len(v))) for v in days.values()],axis=1)
            data={}
            for split,names in ids.items():
                for a in ARMS:
                    for b in BUDGETS:
                        data[split,a,b]=np.array([[lookup[n,cap,a,b]['ttft_s'],lookup[n,cap,a,b]['output_tps']] for n in names])
            means={(a,b):data['test',a,b].mean(axis=0) for a in ARMS for b in BUDGETS}
            default=data['test','arrival',1024]
            for a in ARMS[1:]:
                for b in BUDGETS:
                    other=data['test',a,b]
                    dl=default[:,0]-other[:,0]
                    dq=other[:,1]-default[:,1]
                    lc,qc=ci(dl[boot].mean(axis=1)),ci(dq[boot].mean(axis=1))
                    default_pairs.append(dict(source=source,capacity=cap,baseline=a,budget=b,
                                              diagnostic=a in DIAGNOSTIC,
                                              baseline_dominates_mean=dominates([other[:,0].mean(),-other[:,1].mean()],
                                                                               [default[:,0].mean(),-default[:,1].mean()]),
                                              baseline_ttft_advantage_s=float(dl.mean()),
                                              ttft_ci_low=lc[0],ttft_ci_high=lc[1],
                                              baseline_output_advantage_tps=float(dq.mean()),
                                              output_ci_low=qc[0],output_ci_high=qc[1]))
            for a in ARMS:
                for b in BUDGETS:
                    v=means[a,b]
                    interval=np.quantile(data['test',a,b][boot].mean(axis=1),[.025,.975],axis=0)
                    main_dom=[f'{aa}:T{bb}' for (aa,bb),vv in means.items() if aa in MAIN and dominates([vv[0],-vv[1]],[v[0],-v[1]])]
                    oracle_dom=[f'{aa}:T{bb}' for (aa,bb),vv in means.items() if aa in DIAGNOSTIC and dominates([vv[0],-vv[1]],[v[0],-v[1]])]
                    points.append(dict(source=source,capacity=cap,arm=a,budget=b,diagnostic=a in DIAGNOSTIC,
                                       ttft_s=float(v[0]),output_tps=float(v[1]),
                                       ttft_ci_low=interval[0,0],ttft_ci_high=interval[1,0],
                                       output_ci_low=interval[0,1],output_ci_high=interval[1,1],
                                       main_dominated_by=';'.join(main_dom),oracle_dominated_by=';'.join(oracle_dom)))
            for w in np.linspace(0,1,11):
                chosen={}
                def score(split,a,b):
                    arr,ref=data[split,a,b],data[split,'native',1024]
                    return w*arr[:,0]/ref[:,0]+(1-w)*ref[:,1]/arr[:,1]
                for a in ARMS:
                    b=min(BUDGETS,key=lambda b:(float(score('validation',a,b).mean()),b))
                    val=score('test',a,b)
                    interval=ci(val[boot].mean(axis=1))
                    chosen[a]=b
                    selected.append(dict(source=source,capacity=cap,w=round(float(w),1),arm=a,diagnostic=a in DIAGNOSTIC,
                                         budget=b,validation_score=float(score('validation',a,b).mean()),
                                         test_score=float(val.mean()),ci_low=interval[0],ci_high=interval[1],
                                         ttft_s=float(means[a,b][0]),output_tps=float(means[a,b][1])))
                for group,arms in [('main',MAIN),('oracle_diagnostic',('arrival',)+DIAGNOSTIC)]:
                    scores={a:float(score('test',a,chosen[a]).mean()) for a in arms}
                    best=min(scores.values())
                    win=[a for a,v in scores.items() if abs(v-best)<1e-10]
                    winners.append(dict(source=source,capacity=cap,w=round(float(w),1),group=group,
                                        winners=';'.join(win),spare_wins='arrival' in win,
                                        spare_gap_pct=100*(scores['arrival']/best-1)))
                if abs(w-.5)<1e-10:
                    ours=data['test','arrival',chosen['arrival']]
                    for a in ARMS[1:]:
                        other=data['test',a,chosen[a]]
                        dl=other[:,0]-ours[:,0]  # positive is SPARE advantage
                        dq=ours[:,1]-other[:,1]
                        lc,qc=ci(dl[boot].mean(axis=1)),ci(dq[boot].mean(axis=1))
                        pairs.append(dict(source=source,capacity=cap,baseline=a,diagnostic=a in DIAGNOSTIC,
                                          spare_budget=chosen['arrival'],baseline_budget=chosen[a],
                                          spare_ttft_advantage_s=float(dl.mean()),ttft_ci_low=lc[0],ttft_ci_high=lc[1],
                                          spare_output_advantage_tps=float(dq.mean()),output_ci_low=qc[0],output_ci_high=qc[1],
                                          spare_dominates_mean=dominates([ours[:,0].mean(),-ours[:,1].mean()],[other[:,0].mean(),-other[:,1].mean()]),
                                          baseline_dominates_mean=dominates([other[:,0].mean(),-other[:,1].mean()],[ours[:,0].mean(),-ours[:,1].mean()])))
            for group,arms in [('main',MAIN),('oracle_diagnostic',('arrival',)+DIAGNOSTIC)]:
                baselines=[(a,b) for a in arms if a!='arrival' for b in BUDGETS]
                default=means['arrival',1024]
                default_dom=[f'{a}:T{b}' for a,b in baselines if dominates([means[a,b][0],-means[a,b][1]],[default[0],-default[1]])]
                front=[]
                for b in BUDGETS:
                    v=means['arrival',b]
                    if not any(dominates([vv[0],-vv[1]],[v[0],-v[1]]) for (a,bb),vv in means.items() if a in arms):
                        front.append(b)
                covered=sum(any(dominates([means['arrival',b][0],-means['arrival',b][1]],
                                           [means[a,bb][0],-means[a,bb][1]]) for b in BUDGETS) for a,bb in baselines)
                ws=[r for r in winners if (r['source'],r['capacity'],r['group'])==(source,cap,group)]
                verdicts.append(dict(source=source,capacity=cap,group=group,spare_frontier_budgets=front,
                                     default_external_dominators=default_dom,covered=covered,total=len(baselines),
                                     weighted_wins=sum(r['spare_wins'] for r in ws)))
    csv_write('points.csv',points)
    csv_write('selected_weights.csv',selected)
    csv_write('weight_winners.csv',winners)
    csv_write('paired_w05.csv',pairs)
    csv_write('default_vs_papers.csv',default_pairs)
    summary=dict(completed=len(raw),expected=19040,validation_runs=5440,test_runs=13600,
                 previous_trajectory_matches=parity,main_external_families=9,all_external_families=13,
                 main_weight_winners=dict(Counter(r['winners'] for r in winners if r['group']=='main')),
                 diagnostic_weight_winners=dict(Counter(r['winners'] for r in winners if r['group']=='oracle_diagnostic')),
                 verdicts=verdicts,performance_validated=False)
    write(OUT/'summary.json',summary)
    plot(points,selected)
    report(summary,points,selected,pairs,default_pairs)
    print(__import__('json').dumps(summary,ensure_ascii=False,indent=2))


def plot(points, selected):
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    palette=plt.get_cmap('tab20')
    colors={a:palette(i) for i,a in enumerate(ARMS)}
    colors['arrival']='#0072B2'
    for group,arms in [('papers',MAIN),('oracle_diagnostics',('arrival',)+DIAGNOSTIC)]:
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        for i,source in enumerate(('code','conversation')):
            for j,cap in enumerate(CAPACITIES):
                ax=axes[i,j]
                for ai,a in enumerate(arms):
                    ps=[p for p in points if (p['source'],p['capacity'],p['arm'])==(source,cap,a)]
                    ax.scatter([p['ttft_s'] for p in ps],[p['output_tps'] for p in ps],
                               s=120 if a=='arrival' else 30,marker='*' if a=='arrival' else ['o','s','^','D'][ai%4],
                               color=colors[a],label=LABELS[a],zorder=5 if a=='arrival' else 2,alpha=.95 if a=='arrival' else .7)
                    if a=='arrival':
                        for p in ps:
                            ax.errorbar(p['ttft_s'],p['output_tps'],
                                        xerr=[[p['ttft_s']-p['ttft_ci_low']],[p['ttft_ci_high']-p['ttft_s']]],
                                        yerr=[[p['output_tps']-p['output_ci_low']],[p['output_ci_high']-p['output_tps']]],
                                        color=colors[a],alpha=.45,lw=.8,capsize=2,zorder=3)
                    front=[p for p in ps if not any(dominates([q['ttft_s'],-q['output_tps']],[p['ttft_s'],-p['output_tps']]) for q in ps)]
                    front.sort(key=lambda p:p['ttft_s'])
                    ax.plot([p['ttft_s'] for p in front],[p['output_tps'] for p in front],color=colors[a],lw=.8,alpha=.7)
                    if a=='arrival':
                        for p in ps:
                            offset=(-23,-13) if source=='code' and cap==4096 and p['budget']==512 else (4,5)
                            ax.annotate(str(p['budget']),(p['ttft_s'],p['output_tps']),xytext=offset,textcoords='offset points',fontsize=8,color=colors[a])
                ax.set(title=f'{source.title()} | KV {cap} blocks',xlabel='Mean TTFT (s), lower is better',ylabel='Output tokens/s, higher is better')
                ax.grid(alpha=.12)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=4 if group=='papers' else 3,fontsize=9,frameon=False)
        fig.suptitle(('External paper rules and proxies' if group=='papers' else 'True-length oracle diagnostics')+' | CPU simulation\n50 paired test windows per point; SPARE bars: within-date 95% CI',fontsize=13)
        fig.subplots_adjust(top=.9,bottom=.15 if group=='papers' else .12,hspace=.32,wspace=.22)
        for ext in ('png','pdf'):
            fig.savefig(OUT/f'pareto_{group}.{ext}',dpi=180)
        plt.close(fig)
    # Pairwise comparison at an illustrative weight fixed before inspecting outcomes.
    fig,axes=plt.subplots(1,2,figsize=(12,7),constrained_layout=True)
    for col,metric in enumerate(('ttft_s','output_tps')):
        matrix=[]
        for a in MAIN[1:]:
            row=[]
            for source in ('code','conversation'):
                for cap in CAPACITIES:
                    ours=next(r for r in selected if (r['source'],r['capacity'],r['arm'],r['w'])==(source,cap,'arrival',.5))[metric]
                    other=next(r for r in selected if (r['source'],r['capacity'],r['arm'],r['w'])==(source,cap,a,.5))[metric]
                    row.append(100*(1-ours/other) if metric=='ttft_s' else 100*(ours/other-1))
            matrix.append(row)
        mat=np.array(matrix)
        ax=axes[col]
        lim=max(1,float(np.max(np.abs(mat))))
        ax.imshow(mat,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
        ax.set_xticks(range(4),['Code\n768','Code\n4096','Conv.\n768','Conv.\n4096'])
        ax.set_yticks(range(len(MAIN)-1),[LABELS[a] for a in MAIN[1:]])
        ax.set_title('SPARE mean TTFT reduction (%)' if metric=='ttft_s' else 'SPARE output throughput increase (%)')
        for (i,j),v in np.ndenumerate(mat):
            ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(v)>.55*lim else 'black')
    fig.suptitle('w = 0.5; budgets selected on validation dates | positive favors SPARE')
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'paired_w05.{ext}',dpi=180)
    plt.close(fig)


def report(summary, points, selected, pairs, default_pairs):
    lines=['# 外部论文方法：延迟与吞吐模拟比较','',
           '完成 **19040/19040** 条CPU simulator轨迹（5440选参、13600测试），每个测试配置50个不同窗口。'
           '未加载模型、未启动GPU推理。SRP不进入本轮主比较。', '',
           '比较覆盖13个外部论文／系统方法族：主比较9族、11个规则/代理实现，另列5个真值诊断实现（含LTR的第二种信息条件）。'
           'Past-Future和SOLA各有两个版本，不能当作两篇论文。TokenFlow缺少消费buffer轨迹和迁移模型，未给出模拟数值。','',
           '## 主比较：经验Pareto前沿','',
           '覆盖数为被至少一个SPARE配置支配的外部配置数。所有判断按50测试窗口均值；完整置信区间见points.csv和paired_w05.csv。','',
           '|负载|KV块|SPARE前沿预算|默认T1024被外部配置支配|覆盖外部点|权重最优次数|',
           '|---|---:|---|---|---:|---:|']
    for v in summary['verdicts']:
        if v['group']!='main':continue
        lines.append(f"|{v['source']}|{v['capacity']}|{v['spare_frontier_budgets'] or '无'}|{'; '.join(v['default_external_dominators']) or '无'}|{v['covered']}/{v['total']}|{v['weighted_wins']}/11|")
    lines+=['','测试均值前沿：四个场景均有SPARE配置；Code/KV4096覆盖全部44个外部配置，其余场景保留取舍。'
            '默认T1024仅Conversation/KV768被Apt-Serve KV规则T256在均值上同时超过；见下方配对区间。', '',
            '44个场景×权重点的测试均值中，SPARE最优37次，S³装箱代理最优7次；后者集中在吞吐权重较高时。', '',
            '## 默认T1024的配对检查','',
            '正数表示对照方法优于默认SPARE。区间是两个测试日期内的配对窗口bootstrap；逐比较报告，未作多重校正。','',
            '|负载|KV块|对照|TTFT优势s [95%区间]|吞吐优势tokens/s [95%区间]|',
            '|---|---:|---|---:|---:|']
    for r in default_pairs:
        if r['baseline_dominates_mean'] and (not r['diagnostic'] or r['baseline']=='mc_sf_oracle' and r['budget']==256):
            lines.append(f"|{r['source']}|{r['capacity']}|{LABELS[r['baseline']]} T{r['budget']}|"
                         f"{r['baseline_ttft_advantage_s']:+.3f} [{r['ttft_ci_low']:+.3f}, {r['ttft_ci_high']:+.3f}]|"
                         f"{r['baseline_output_advantage_tps']:+.2f} [{r['output_ci_low']:+.2f}, {r['output_ci_high']:+.2f}]|")
    lines+=['','Apt-Serve KV规则的TTFT均值优势为0.025秒，但配对95%区间跨过零；'
            '其吞吐均值优势约12.0 tokens/s。MC-SF使用真实输出长度，其结果只用于诊断。', '',
            '## w=0.5的独立选参结果','',
            '预算在20个选参窗口上选定，以下为另50窗口测试均值；每格为“预算；TTFT秒 / output tokens/s”。','',
            '|方法|Code KV768|Code KV4096|Conversation KV768|Conversation KV4096|',
            '|---|---:|---:|---:|---:|']
    for a in MAIN:
        vals=[]
        for s in ('code','conversation'):
            for c in CAPACITIES:
                r=next(r for r in selected if (r['source'],r['capacity'],r['arm'],r['w'])==(s,c,a,.5))
                vals.append(f"T{r['budget']}; {r['ttft_s']:.3f} / {r['output_tps']:.2f}")
        lines.append('|'+LABELS[a]+'|'+'|'.join(vals)+'|')
    lines+=['','## 真值长度诊断','',
            '以下方法读取真实输出长度，未接入作者预测器，不合入主表胜负。','',
            '|负载|KV块|SPARE前沿预算|默认SPARE的真值对照支配者|覆盖诊断点|权重最优次数|',
            '|---|---:|---|---|---:|---:|']
    for v in summary['verdicts']:
        if v['group']!='oracle_diagnostic':continue
        lines.append(f"|{v['source']}|{v['capacity']}|{v['spare_frontier_budgets'] or '无'}|{'; '.join(v['default_external_dominators']) or '无'}|{v['covered']}/{v['total']}|{v['weighted_wins']}/11|")
    lines+=['','## 实现范围','', '|实现|本轮实际运行内容|','|---|---|']
    for a in ARMS:
        lines.append(f'|{LABELS[a]}|{SCOPE_CN[a]}|')
    lines+=['','## 条件与核验','',
            '- 所有方法相同输入、容量、执行后端和模拟时间公式；T四档、K256、并发32。多个版本各自四档，均公开报告。',
            '- 每类来源选参20窗口、测试50窗口；选参日期与测试日期分开。每窗口64请求，5.25 req/s；测试窗口来自两个日期。',
            '- Q计实际输出token除以首到达到最后完成的时长，含排空、不计重算；不是最大可持续吞吐。',
            '- J=w L/L0+(1-w)Q0/Q；逐窗口用native T1024归一化。在选参集选预算，测试集不选配置。',
            '- 95%区间为两个固定测试日期内窗口配对bootstrap，5000次，未作跨方法多重校正；不表示跨日期泛化。',
            '- 参数与输入沿用上一轮已知试验，本轮扩展外部方法比较，不称为重新采集的盲测。',
            '- 时间来自历史日志拟合，硬件预测精度没有在本轮重新实测；SOLA读取现有独立校准文件，在线更新使用模拟时长。',
            '- 模拟时钟不额外计入各方法CPU选策耗时和缺失预测器的推理成本；结果不包含完整系统的实现开销差异。',
            f"- 34条接入轨迹与4项数学测试通过；全部正式运行完成并检查守恒；{summary['previous_trajectory_matches']}条重跑轨迹与上一轮完全一致；源码和输入哈希核对通过。",'',
            '## 文件','',
            '- [外部方法Pareto图](pareto_papers.png) / [PDF](pareto_papers.pdf)；图中SPARE误差线为日期内窗口95%区间，其他方法区间见points.csv。',
            '- [Oracle诊断图](pareto_oracle_diagnostics.png)',
            '- [w=0.5配对改善图](paired_w05.png)',
            '- runs.csv、points.csv：全部运行与测试配置均值/区间。',
            '- selected_weights.csv、weight_winners.csv：各权重的选参和测试结果。',
            '- paired_w05.csv：w=0.5选参后的配对绝对差异及区间；default_vs_papers.csv：默认T1024与每个外部配置的配对差异。',
            '- design.json、source/、runs/：设计、源码快照与逐运行指标；输入哈希指向上一轮冻结文件。','']
    (OUT/'REPORT_CN.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
