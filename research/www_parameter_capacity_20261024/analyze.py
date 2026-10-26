"""Audit capacity endpoints and export a single table with matching KV and latency."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import statistics

from . import run

DEST = run.ROOT / 'writing/www/manuscript/sensitivity_capacity_20261024'
LABELS = {'spare': 'Default', 'n0_4': 'n0 = 4', 'n0_16': 'n0 = 16',
          'alpha64': 'alpha = 64', 'alpha256': 'alpha = 256',
          'H05': 'H = 0.5 s', 'H2': 'H = 2 s',
          'tau05': 'tau = 0.5 s', 'tau2': 'tau = 2 s'}
TEX = {'spare': r'\textbf{Default}', 'n0_4': '$n_0=4$', 'n0_16': '$n_0=16$',
       'alpha64': r'$\alpha=64$', 'alpha256': r'$\alpha=256$',
       'H05': '$H=0.5$ s', 'H2': '$H=2$ s',
       'tau05': r'$\tau=0.5$ s', 'tau2': r'$\tau=2$ s'}


def csvwrite(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    samples=run.prepare();DEST.mkdir(parents=True,exist_ok=True)
    groups={};endpoints=[];rates={};records=[]
    for source in run.SOURCES:
        wanted={s['name'] for s in samples if s['source']==source}
        for kv in run.KVS:
            for arm in run.ARMS:
                path=run.OUT/'groups'/f'{source}_c{kv}_{arm}.json'
                g=run.READ(path);groups[source,kv,arm]=g;scans={}
                for qps in g['rates']:
                    p=run.rate_path(source,kv,arm,qps);d=run.READ(p)
                    assert len(d['rows'])==24 and {r['name'] for r in d['rows']}==wanted
                    assert all(r['source']==source and r['capacity']==kv and r['rate']==qps for r in d['rows'])
                    assert all(r['arm'] in (('spare','arrival') if arm=='spare' else (arm,)) for r in d['rows'])
                    summary=run.confirm.summary(d['rows'],qps)
                    assert summary==d['summary']
                    rates[source,kv,arm,qps]=d;scans[qps]=summary
                    records.append(dict(path=str(p.relative_to(run.ROOT)),sha256=run.DIGEST(p)))
                for theta in run.THETAS:
                    c=g['capacities'][f'{theta:g}']
                    low,high,causes=run.pilot.old.bounds(scans,theta)
                    assert (low,high,causes)==(c['last_pass_qps'],c['first_fail_qps'],c['failure_causes'])
                    assert 0<high-low<=.0200001
                    for qps,expected in ((low,True),(high,False)):
                        assert run.passed(scans[qps],qps,theta)==expected
                        endpoints.append(dict(source=source,kv_blocks=kv,arm=arm,theta_s=theta/1000,
                                              qps=qps,expected_pass=expected,verified=True))
    reference={}
    for source in run.SOURCES:
        for kv in run.KVS:
            mincap=min(groups[source,kv,a]['capacities']['2000']['last_pass_qps'] for a in run.ARMS)
            good=[q for q in run.REFERENCE_RATES if q<=mincap and all(
                run.passed(rates[source,kv,a,q]['summary'],q,2000.) for a in run.ARMS)]
            assert good,(source,kv,mincap)
            reference[source,kv]=max(good)
    combined=[];window_rows=[]
    for source in run.SOURCES:
        for kv in run.KVS:
            qps=reference[source,kv]
            default=rates[source,kv,'spare',qps]['summary']
            for arm in run.ARMS:
                data=rates[source,kv,arm,qps];s=data['summary']
                for r in data['rows']:
                    window_rows.append(dict(r,source=source,kv_blocks=kv,arm=arm,qps=qps))
                for theta in run.THETAS:
                    cap=groups[source,kv,arm]['capacities'][f'{theta:g}']
                    dc=groups[source,kv,'spare']['capacities'][f'{theta:g}']['last_pass_qps']
                    combined.append(dict(source=source,kv_blocks=kv,arm=arm,parameter=LABELS[arm],
                        theta_s=theta/1000,capacity_qps=cap['last_pass_qps'],
                        first_fail_qps=cap['first_fail_qps'],resolution_qps=cap['resolution_qps'],
                        failure_causes=';'.join(cap['failure_causes']),matched_qps=qps,
                        mean_ttft_ms=s['mean_ttft_ms'],mean_p99_tbt_ms=s['p99_tbt_ms'],
                        mean_backlog_drift=s['backlog_drift'],windows=24,
                        mean_preemptions=s['preemptions'],mean_recomputed_tokens=s['recomputed_tokens'],
                        mean_interventions=(statistics.fmean(r['interventions'] for r in data['rows'])
                                            if all('interventions' in r for r in data['rows']) else ''),
                        mean_rounds=(statistics.fmean(r['rounds'] for r in data['rows'])
                                     if all('rounds' in r for r in data['rows']) else ''),
                        capacity_change_pct=100*(cap['last_pass_qps']/dc-1),
                        ttft_change_pct=100*(s['mean_ttft_ms']/default['mean_ttft_ms']-1)))
    csvwrite(DEST/'combined.csv',combined)
    csvwrite(DEST/'matched_window_metrics.csv',window_rows)
    csvwrite(DEST/'endpoint_audit.csv',endpoints)
    index={(r['source'],r['kv_blocks'],r['arm'],r['theta_s']):r for r in combined}
    md=['**参数敏感性：同 KV 的容量与延迟**','',
        'Code、Conversation 各24个冻结配对窗口，Poisson到达60 s，预热20 s；'
        '窗口mean TTFT、P99 TBT与积压按既定协议聚合。所有结果均为GPU-fitted CPU模拟。','',
        '每个KV下容量和延迟使用相同策略、输入窗口及KV配置；延迟在所有参数变体共同的QPS下比较。'
        '默认 n0=8、alpha=128、H=tau=1 s；每行仅改变所列参数。','']
    for theta in (2.,4.):
        md += [f'**mean-TTFT limit = {theta:g} s**','',
               '| Workload / parameter | KV1024 capacity ↑ | TTFT ↓ | KV2048 capacity ↑ | TTFT ↓ | KV4096 capacity ↑ | TTFT ↓ |',
               '|---|---:|---:|---:|---:|---:|---:|']
        tex=[r'\begin{table*}[t]',r'\centering',r'\small',
             r'\caption{Parameter sensitivity under a '+f'{theta:g}'+r'-s mean-TTFT limit. '
             r'Capacity (C, QPS) and matched-load mean TTFT (L, ms) use identical KV settings and paired windows. '
             r'Bold identifies the default configuration.}',
             r'\label{tab:parameter-sensitivity-'+f'{theta:g}'+r's}',
             r'\begin{tabular}{lrrrrrr}',r'\toprule',
             r'& \multicolumn{2}{c}{KV=1,024} & \multicolumn{2}{c}{KV=2,048} & \multicolumn{2}{c}{KV=4,096} \\',
             r'Configuration & C$\uparrow$ & L$\downarrow$ & C$\uparrow$ & L$\downarrow$ & C$\uparrow$ & L$\downarrow$ \\',r'\midrule']
        for source in run.SOURCES:
            qs='/'.join(f'{reference[source,k]:g}' for k in run.KVS)
            md.append(f'| **{source.title()} — TTFT at {qs} QPS** | | | | | | |')
            tex.append(r'\multicolumn{7}{l}{\textit{'+source.title()+f'; TTFT at {qs} QPS for the three KV settings'+r'}} \\')
            for arm in run.ARMS:
                cells=[]
                for kv in run.KVS:
                    r=index[source,kv,arm,theta];cells += [f"{r['capacity_qps']:.2f}",f"{r['mean_ttft_ms']:.1f}"]
                label=LABELS[arm]
                if arm=='spare':label=f'**{label}**'
                md.append('| '+label+' | '+' | '.join(cells)+' |')
                tex.append(TEX[arm]+' & '+' & '.join(cells)+r' \\')
            tex.append(r'\midrule')
        tex[-1]=r'\bottomrule'
        tex += [r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{0.96\textwidth}\footnotesize',
                r'Each workload uses 24 paired windows. Capacity additionally requires mean window P99 TBT '
                r'$\leq200$ ms and mean backlog growth $\leq\max(2,\lambda)$. '
                r'All variants use the same offered QPS within each workload/KV setting. '
                r'Capacity boundary intervals have width 0.01--0.02 QPS; these are search resolutions, '
                r'not confidence intervals. Defaults: $n_0=8$, $\alpha=128$, $H=\tau=1$ s.',
                r'\end{minipage}',r'\end{table*}']
        (DEST/f'table_{theta:g}s.tex').write_text('\n'.join(tex)+'\n')
        md += ['', '容量单位 QPS，TTFT 单位 ms。容量边界分辨率为0.01–0.02 QPS，不是统计置信区间。','']
    (DEST/'TABLE_CN.md').write_text('\n'.join(md)+'\n')
    # Browser-readable rendering of exactly the same values, without a plotting dependency.
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>SPARE parameter sensitivity</title>',
          '<style>body{font:15px Arial,sans-serif;margin:36px;color:#202020}table{border-collapse:collapse;margin:20px 0 40px;width:100%;max-width:1050px}th,td{padding:8px 14px;text-align:right}th:first-child,td:first-child{text-align:left}thead{border-top:2px solid;border-bottom:1px solid}.source{background:#f1f4f7;font-weight:bold}.default{font-weight:bold}tbody{border-bottom:2px solid}p{max-width:1000px;line-height:1.5}</style>',
          '<h1>Parameter sensitivity</h1><p>Capacity and mean TTFT use identical KV settings and 24 paired windows per workload. CPU simulation with GPU-fitted timing. Each row changes one parameter; bold marks the default.</p>']
    for theta in (2.,4.):
        body += [f'<h2>Capacity under a {theta:g}-s mean-TTFT limit</h2>',
          '<table><thead><tr><th></th><th colspan="2">KV = 1,024</th><th colspan="2">KV = 2,048</th><th colspan="2">KV = 4,096</th></tr><tr><th>Parameter</th>'+3*'<th>Capacity ↑ (QPS)</th><th>TTFT ↓ (ms)</th>'+'</tr></thead><tbody>']
        for source in run.SOURCES:
            qs=' / '.join(f'{reference[source,k]:g}' for k in run.KVS)
            body.append(f'<tr class="source"><td colspan="7">{source.title()}: TTFT measured at {qs} QPS respectively</td></tr>')
            for arm in run.ARMS:
                cls='default' if arm=='spare' else ''
                cells=[]
                for kv in run.KVS:
                    r=index[source,kv,arm,theta];cells += [f"{r['capacity_qps']:.2f}",f"{r['mean_ttft_ms']:.1f}"]
                body.append(f'<tr class="{cls}"><td>{html.escape(LABELS[arm])}</td>'+''.join('<td>'+x+'</td>' for x in cells)+'</tr>')
        body.append('</tbody></table>')
    body.append('<p>Additional capacity constraints: mean window P99 TBT ≤ 200 ms; mean backlog growth ≤ max(2, QPS). Capacity search resolution: 0.01–0.02 QPS, not a confidence interval. Defaults: n0 = 8, alpha = 128, H = tau = 1 s.</p></html>')
    (DEST/'table.html').write_text('\n'.join(body)+'\n')
    allrows=[r for d in rates.values() for r in d['rows']]
    audit=dict(groups=len(groups),expected_groups=54,capacity_endpoints_verified=len(endpoints),
               rate_files=len(rates),window_records=len(allrows),
               reused_windows=sum(d['reused_rows'] for d in rates.values()),
               new_windows=sum(d['newly_simulated'] for d in rates.values()),
               invalid_windows=sum(not r['valid'] for r in allrows),
               reference_qps={f'{s}_c{k}':v for (s,k),v in reference.items()},
               output_rows=len(combined),gpu_execution=False,
               analysis_sha256=run.DIGEST(Path(__file__)),
               design_sha256=run.DIGEST(run.OUT/'design.json'),
               complete=True,source_rate_files=records)
    run.WRITE(DEST/'audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='source_rate_files'},indent=2))


if __name__=='__main__':main()
