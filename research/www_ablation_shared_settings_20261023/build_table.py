"""Combine capacity and matched-load mean TTFT for the representative table."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/www_ablation_shared_settings_20261023_r1'
PRIMARY = ROOT / 'artifacts/www_component_ablation_mean_20261020_r4'
ARMS = ('spare', 'srp', 'constant8', 'no_arrival')
LABELS = ('Full SPARE', 'SRP', 'Fixed n=8', 'tau=0')
KVS = (1024, 3584, 5120)
COARSE = (.25, .5, .75, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 8., 10.)


def read(path):
    return json.loads(path.read_text())


def build():
    capacities = {(int(r['kv_blocks']), r['arm']): float(r['capacity_qps'])
                  for r in csv.DictReader((OUT/'analysis/capacities.csv').open())
                  if float(r['theta_s']) == 4}
    wanted = {s['name'] for s in read(OUT/'design.json')['samples']}
    rates = {kv: max(q for q in COARSE if q <= min(capacities[kv, a] for a in ARMS)) for kv in KVS}
    rows = []
    for kv in KVS:
        archive = PRIMARY if kv == 1024 else OUT
        for arm in ARMS:
            path = archive/'rates'/f'conversation_c{kv}_{arm}_r{rates[kv]:g}.json'
            raw = read(path)['rows']
            assert {r['name'] for r in raw} == wanted and all(r['valid'] for r in raw)
            means = {k: sum(r[k] for r in raw)/len(raw) for k in
                     ('mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift')}
            assert means['mean_ttft_ms'] <= 4000 and means['p99_tbt_ms'] <= 200
            assert means['backlog_drift'] <= max(2, rates[kv])
            rows.append(dict(kv_blocks=kv, arm=arm, capacity_qps=capacities[kv, arm],
                             capacity_ttft_limit_s=4, ttft_offered_qps=rates[kv],
                             **means, windows=24, source_file=str(path.relative_to(ROOT))))
    with (OUT/'analysis/representative_combined.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lookup = {(r['kv_blocks'], r['arm']): r for r in rows}
    lines = ['| 方法 | 1024 容量↑ | 1024 TTFT↓ | 3584 容量↑ | 3584 TTFT↓ | 5120 容量↑ | 5120 TTFT↓ |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm, label in zip(ARMS, LABELS):
        values = [value for kv in KVS for value in
                  (f"{lookup[kv, arm]['capacity_qps']:.2f}", f"{lookup[kv, arm]['mean_ttft_ms']:.0f}")]
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    caption = ('容量单位为QPS，mean TTFT单位为ms。容量采用mean TTFT≤4 s及相同TBT/积压约束；'
               'TTFT分别在KV=1024/3584/5120的1.5/3/4 QPS下测量，同一KV下四种方法负载相同。'
               '到达率取该档所有方法容量范围内最高的已有粗扫描点，12个单元均满足性能约束。')
    table = '\n'.join(lines)+'\n\n'+caption+'\n'
    (OUT/'analysis/representative_combined.md').write_text(table)
    tex = [r'\begin{table*}[t]', r'\centering',
           r'\caption{Component ablation on Conversation. Capacity (QPS) uses a 4-s mean-TTFT limit and the same TBT and backlog constraints. Mean TTFT (ms) is measured at a common arrival rate within each KV setting: 1.5, 3, and 4 QPS, respectively. All variants use the same 24 test windows.}',
           r'\label{tab:component-ablation}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
           r'& \multicolumn{2}{c}{KV=1024} & \multicolumn{2}{c}{KV=3584} & \multicolumn{2}{c}{KV=5120} \\',
           r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
           r'Variant & Capacity $\uparrow$ & Mean TTFT $\downarrow$ & Capacity $\uparrow$ & Mean TTFT $\downarrow$ & Capacity $\uparrow$ & Mean TTFT $\downarrow$ \\', r'\midrule']
    tex_labels = ('Full SPARE', 'SRP', r'Fixed $n=8$', r'$\tau=0$')
    for arm, label in zip(ARMS, tex_labels):
        values = [v for kv in KVS for v in
                  (f"{lookup[kv, arm]['capacity_qps']:.2f}", f"{lookup[kv, arm]['mean_ttft_ms']:.0f}")]
        tex.append(label+' & '+' & '.join(values)+r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (OUT/'analysis/representative_combined.tex').write_text('\n'.join(tex)+'\n')
    note_path = OUT/'analysis/selection_note.md'
    note = note_path.read_text()
    start = note.index('## 推荐主表：')
    end = note.index('## 每部分应写什么', start)
    note = note[:start]+'## 推荐主表：容量与同负载 mean TTFT\n\n'+table+'\n'+note[end:]
    note_path.write_text(note)
    print(table)
    for kv, arm in [(1024, 'no_arrival'), (3584, 'srp'), (5120, 'constant8')]:
        print(kv, arm, 'TTFT reduction %', 100*(1-lookup[kv,'spare']['mean_ttft_ms']/lookup[kv,arm]['mean_ttft_ms']))


if __name__ == '__main__':
    build()
