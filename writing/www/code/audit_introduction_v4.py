"""Recompute Introduction V4 claims from archived GPU results; no experiments run."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/www_introduction_revision_20261010_r1"
G2 = ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1"
METRICS = ("ttft_ms_mean", "token_itl_ms_p99", "tpot_ms_mean", "output_throughput_tps")
REFERENCES = ("srp", "native", "sarathi", "past_future")


def summarize(rows, metric, reference):
    by = defaultdict(list)
    for row in rows:
        if row["arm"] in ("arrival", reference):
            by[row["cell"], row["arm"]].append(float(row[metric]))
    effects = []
    for cell, arm in sorted(by):
        if arm != "arrival":
            continue
        ours, ref = by[cell, "arrival"], by[cell, reference]
        assert len(ours) == len(ref) == 4, (cell, reference)
        a, b = st.fmean(ours), st.fmean(ref)
        assert b > 0
        effect = 100 * (a / b - 1) if metric == "output_throughput_tps" else 100 * (1 - a / b)
        effects.append(dict(cell=cell, ours=a, reference_value=b, improvement_pct=effect))
    return dict(metric=metric, reference=reference, cells=len(effects),
                mean_improvement_pct=st.fmean(e["improvement_pct"] for e in effects),
                maximum=max(effects, key=lambda e: e["improvement_pct"]),
                wins=sum(e["improvement_pct"] > 0 for e in effects))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g2 = []
    for row in csv.DictReader((G2 / "analysis/runs.csv").open()):
        if row["arm"] not in ("arrival", *REFERENCES):
            continue
        raw = (G2 / row["result_path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        result = json.loads(raw)
        assert result["status"] == "complete"
        gaps = [right - left for clock in result["clocks"].values()
                for left, right in zip(clock["token_times_ms"], clock["token_times_ms"][1:])]
        p99 = st.quantiles(gaps, n=100, method="inclusive")[98]
        assert abs(p99 - result["metrics"]["token_itl_ms_p99"]) < 1e-8
        row["token_itl_ms_p99"] = p99
        g2.append(row)
    assert len(g2) == 36 * 5 * 4
    effects = [summarize(g2, metric, ref) for metric in METRICS for ref in REFERENCES]
    totals = {}
    for field in ("preemptions", "recomputed_tokens"):
        a = sum(float(r[field]) for r in g2 if r["arm"] == "arrival")
        b = sum(float(r[field]) for r in g2 if r["arm"] == "srp")
        totals[field] = dict(ours=a, srp=b, reduction_pct=100 * (1 - a / b))
    chat_dir = OUT / "remote/www_chat_only_gpu_20261009_r1"
    chat = list(csv.DictReader((chat_dir / "analysis/runs.csv").open()))
    remote = json.loads((chat_dir / "analysis/summary.json").read_text())
    assert remote["audited"] and remote["groups"] == 18 and remote["runs"] == len(chat) == 288
    chat_effects = [summarize(chat, "ttft_ms_mean", ref) for ref in ("srp", "fixed256", "fixed512")]
    for effect in chat_effects:
        expected = next(r for r in remote["relative_to_arrival"] if r["reference"] == effect["reference"])
        assert effect["cells"] == 18
        assert abs(effect["mean_improvement_pct"] - expected["arrival_mean_ttft_improvement_pct"]) < 1e-9
    report = dict(g2_verified_raw_results=len(g2), tbt_p99_recomputed_from_token_timestamps=True,
                  g2=effects, g2_totals=totals,
                  chat=chat_effects, chat_validation="Transport hashes and arithmetic checked locally; full round audit reported by remote analysis.")
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for effect in effects:
        m = effect["maximum"]
        print(effect["metric"], effect["reference"], "mean", round(effect["mean_improvement_pct"], 2),
              "max", round(m["improvement_pct"], 2), m["cell"], "values", round(m["ours"], 3), round(m["reference_value"], 3))
    print("CHAT", [(e["reference"], round(e["mean_improvement_pct"], 2)) for e in chat_effects])


if __name__ == "__main__":
    main()
