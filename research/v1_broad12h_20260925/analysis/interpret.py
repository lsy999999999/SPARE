"""Post-result descriptive summaries, preserving the preregistered primary metric."""
import csv,json,statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'artifacts/v1_broad12h_20260925_r1'
rows=list(csv.DictReader((OUT/'analysis/comparisons.csv').open()));real=[r for r in rows if r['family']=='measured_lengths']
groups=sorted({r['group'] for r in real})
zero=[g for g in groups if all(float(r['interventions'])==0 for r in real if r['group']==g)]
result=dict(scope='Descriptive follow-up after complete frozen matrix; thresholds 2% and 5% are descriptive, not statistical significance tests.',zero_intervention_groups=zero,methods={})
old=[r for r in real if r['arm']=='old']
for arm in ('arrival','combined','pressure_decode','pressure_constant'):
    rr=[r for r in real if r['arm']==arm];gains=[float(r['gain_old_pct']) for r in rr];spread=sorted(float(r['ttft_spread_pct']) for r in rr)
    result['methods'][arm]=dict(cells=len(rr),mean_gain_old_pct=st.fmean(gains),median_gain_old_pct=st.median(gains),
        mean_gain_arrival_pct=st.fmean(float(r['gain_arrival_pct']) for r in rr),
        best_old_pct=max(gains),worst_old_pct=min(gains),win_over2=sum(v>2 for v in gains),loss_over2=sum(v< -2 for v in gains),
        win_over5=sum(v>5 for v in gains),loss_over5=sum(v< -5 for v in gains),
        both_repeats_win_over2=sum(all(float(r[f'gain_old_r{i}_pct'])>2 for i in (0,1)) for r in rr),
        both_repeats_loss_over2=sum(all(float(r[f'gain_old_r{i}_pct'])< -2 for i in (0,1)) for r in rr),
        median_repeat_spread=st.median(spread),max_repeat_spread=max(spread),repeat_spread_over5=sum(v>5 for v in spread),
        pooled_mean_ttft_gain_pct=100*(1-st.fmean(float(r['ttft_ms']) for r in rr)/st.fmean(float(r['ttft_ms']) for r in old)))
(OUT/'analysis/interpretation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['methods'],indent=2))
