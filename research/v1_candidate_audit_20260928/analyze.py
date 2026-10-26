"""Audit every run; separate discovery minima from fresh confirmation gains."""
import csv
import gzip
import statistics as st
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_candidate_audit_20260928.common import *
from research.v1_candidate_audit_20260928.audit import audit
from research.v1_candidate_audit_20260928.run import verify


def load_rounds(run):
    path=OUT/run['rounds_path'];assert digest(path)==run['rounds_sha256']
    with gzip.open(path,'rt') as f:return json.load(f)


def csv_write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    d=verify();checks=[];states=[];actions=[];contrasts=[];excluded=[]
    for group in d['groups']:
        folder=OUT/'groups'/group['key']
        if not (folder/'status.json').exists() or read(folder/'status.json')['state']!='complete' or (folder/'exclusion.json').exists():
            excluded.append(group['key']);continue
        assert read(folder/'aa.json')['passed']
        anchor_run=read(folder/'runs/anchor.json');anchor=load_rounds(anchor_run)
        runs={}
        for path in sorted((folder/'runs').glob('*.json')):
            run=read(path);rows=load_rounds(run)
            check=audit(run,rows,anchor);checks.append(dict(group=group['key'],run=path.stem,**check))
            if 'action' in run:
                runs[(run['target'],run['phase'],run['action']['id'],run['repeat'])]=run
                actions.append(dict(group=group['key'],target=run['target'],phase=run['phase'],repeat=run['repeat'],
                    action=run['action']['id'],budget=run['action']['budget'],chunk=run['action']['chunk'],
                    aliases=json.dumps(run['action']['aliases']),families='/'.join(run['action']['families']),
                    **{k:v for k,v in run['label'].items() if k not in ('first_order','first_release','first_preemption')}))
        selected=read(folder/'selected.json')['states']
        for target in selected:
            index=target['round'];winner=read(folder/f's{index}_winners.json');w=winner['winners']
            def costs(phase,aid):return [runs[(index,phase,aid,r)]['label']['area_per_request_ms'] for r in range(2)]
            cur,old,new=[costs('confirm',w[k]) for k in ('current','best_old','best_expanded')]
            eg=[a-b for a,b in zip(old,new)];sg=[a-b for a,b in zip(cur,old)]
            total=[a-b for a,b in zip(cur,new)]
            allcost={a['id']:costs('discovery',a['id']) for a in target['actions']}
            oldids=[a['id'] for a in target['actions'] if 'old' in a['families'] or 'base' in a['families']]
            assert min(allcost,key=lambda a:(st.fmean(allcost[a]),a))==w['best_expanded']
            assert min(oldids,key=lambda a:(st.fmean(allcost[a]),a))==w['best_old']
            # A repeat-based scale, not a significance test; avoid claiming tiny minima.
            noise=max(max(v)-min(v) for v in (cur,old,new))
            confirmed=all(g>max(noise,1.) for g in total)
            def desc(aid):
                a=next(a for a in target['actions'] if a['id']==aid)
                return f"({a['budget']},{a['chunk']})"
            states.append(dict(group=group['key'],target=index,base_T=group['budget'],base_K=group['chunk'],
                reason=target['reason'],k_effective=target['k_effective'],k_matched=target['k_matched'],
                plans=len(target['actions']),new_plans=target['new_plans'],current=desc(w['current']),
                best_old=desc(w['best_old']),best_expanded=desc(w['best_expanded']),
                discovery_expansion_gain=st.fmean(allcost[w['best_old']])-st.fmean(allcost[w['best_expanded']]),
                confirm_expansion_gain=st.fmean(eg),confirm_selection_gain=st.fmean(sg),confirm_total_gain=st.fmean(total),
                total_r0=total[0],total_r1=total[1],expansion_r0=eg[0],expansion_r1=eg[1],
                confirmation_range=noise,directional_confirmed=confirmed))
            for i,a in enumerate(target['actions']):
                for b in target['actions'][i+1:]:
                    pairs=[(x,y) for x in a['aliases'] for y in b['aliases'] if x[0]==y[0] and x[1]!=y[1]]
                    if not pairs:continue
                    delta=[x-y for x,y in zip(allcost[a['id']],allcost[b['id']])]
                    contrasts.append(dict(group=group['key'],target=index,a=a['id'],b=b['id'],
                        parameter_pairs=json.dumps(pairs),equal_q=a['q']==b['q'],equal_growth=a['growth']==b['growth'],
                        a_minus_b_ms=st.fmean(delta),r0=delta[0],r1=delta[1],
                        note='Discovery K contrast; selected examples need confirmation'))
    csv_write(OUT/'analysis/actions.csv',actions);csv_write(OUT/'analysis/states.csv',states)
    csv_write(OUT/'analysis/k_contrasts.csv',contrasts)
    write(OUT/'analysis/audit.json',dict(passed=True,episodes=len(checks),rounds=sum(c['rounds'] for c in checks),excluded=excluded,checks=checks))
    write(OUT/'analysis/summary.json',dict(states=states,accepted_groups=len(d['groups'])-len(excluded),excluded=excluded))
    lines=['# 二维候选审计结果','',f'有效组 {len(d["groups"])-len(excluded)}/{len(d["groups"])}；状态 {len(states)}；审计 {len(checks)} 次运行、{sum(c["rounds"] for c in checks)} 轮。',
        '','正值为等待减少，单位为剩余首token等待面积/64（ms）。选择赢家使用发现阶段，收益使用新的确认阶段。',
        '','|组|轮|合法计划|current|旧池赢家|扩展池赢家|扩池收益|旧池选序收益|总收益|两次总收益|',
        '|---|---:|---:|---|---|---|---:|---:|---:|---|']
    for s in states:
        lines.append(f"|{s['group']}|{s['target']}|{s['plans']}|{s['current']}|{s['best_old']}|{s['best_expanded']}|{s['confirm_expansion_gain']:+.3f}|{s['confirm_selection_gain']:+.3f}|{s['confirm_total_gain']:+.3f}|{s['total_r0']:+.3f}/{s['total_r1']:+.3f}|")
    lines+=['','有限候选池、复用开发窗口、同组同状态单轮动作实验；不代表闭环策略收益或全局最优。跨基础配置状态不同，不能将差值全部归给比例或绝对数。',
        '','K对照全部保存在analysis/k_contrasts.csv；含参数别名，避免把不改变执行计划的K变化重复计数。',
        '',f'缺失/排除：{excluded}']
    (OUT/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(accepted_groups=len(d['groups'])-len(excluded),states=len(states),episodes=len(checks),rounds=sum(c['rounds'] for c in checks)),ensure_ascii=False))


if __name__=='__main__':main()
