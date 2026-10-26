"""Independent paper panels for accepted A/B and capacity/load measurements."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import random
import statistics as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "writing/www/manuscript/latex/figures/evaluation"
ABC = ROOT / "artifacts/www_plan_abc_20261008_r1"
G4 = ROOT / "artifacts/www_capacity_load_20261007_r1"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def main():
    paired = rows(ABC / "a/paired.csv")
    windows = sorted({r["window"] for r in paired})
    assert len(paired) == 36 and len(windows) == 6
    fig, ax = plt.subplots(figsize=(3.35, 1.95), layout="constrained")
    for j, capacity in enumerate([768, 2048, 4096]):
        for k, shape in enumerate(["steady", "burst_stop"]):
            rr = [r for r in paired if int(r["capacity"]) == capacity and r["shape"] == shape]
            ax.scatter([float(r["improvement_pct"]) for r in rr],
                       [windows.index(r["window"])+(j*2+k-2.5)*.11 for r in rr],
                       color=["#195c9b", "#238a83", "#c29342"][j],
                       marker="o" if k == 0 else "^", s=16,
                       label=f"{capacity}, {'steady' if k == 0 else 'burst'}")
    ax.axvline(0, color=".5", lw=.7, ls="--")
    ax.set_yticks(range(6), ["Code 0", "Code 1", "Code 2", "Conv. 0", "Conv. 1", "Conv. 2"])
    ax.invert_yaxis()
    ax.set_xlabel("SPARE TTFT reduction over SRP (%)")
    ax.grid(axis="x", alpha=.15)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, 1), fontsize=6.3,
              columnspacing=.5, handletextpad=.3, borderaxespad=.2)
    save(fig, "a_all_scenarios")

    states = rows(ABC / "b_v2/analysis/states.csv")
    assert len(states) == 24
    fig, ax = plt.subplots(figsize=(3.35, 1.95), layout="constrained")
    for i, r in enumerate(states):
        value = float(r["mean_benefit_ms"])
        ax.errorbar(i, value, yerr=[[value-float(r["min_benefit_ms"])],
                                    [float(r["max_benefit_ms"])-value]],
                    fmt="o" if r["capacity"] == "768" else "s",
                    color="#195c9b" if value >= 0 else "#b5576e",
                    ms=3, capsize=2, lw=.8)
    ax.axhline(0, color=".5", lw=.7, ls="--")
    for x in [3.5, 7.5, 11.5, 15.5, 19.5]:
        ax.axvline(x, color=".85", lw=.6)
    ax.set_xticks([1.5+4*i for i in range(6)], ["Code 0", "Code 1", "Code 2", "Conv. 0", "Conv. 1", "Conv. 2"],
                  rotation=25, ha="right")
    ax.set_ylabel("Net waiting reduction\n(ms/request)")
    ax.grid(axis="y", alpha=.15)
    save(fig, "b_all_states")

    agg = {r["arm"]: r for r in rows(ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1/analysis/aggregate.csv")}
    arms = ["arrival", "native", "sarathi", "past_future", "srp"]
    colors = ["#195c9b", "#658797", "#c29342", "#238a83", "#ab72a4"]
    for metric, name, unit in [
        ("output_throughput_tps", "service_throughput", "Output tokens/s"),
        ("slo_ttft5000_maxitl200_goodput_rps", "service_goodput", "Goodput (req/s)")]:
        fig, ax = plt.subplots(figsize=(1.63, 1.6), layout="constrained")
        ax.barh(range(5), [float(agg[a][metric]) for a in arms], color=colors, height=.65)
        ax.set_yticks(range(5), ["SPARE", "V1", "Sarathi", "Past-Future", "SRP"])
        ax.invert_yaxis()
        ax.set_xlabel(unit, fontsize=7)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        save(fig, name)

    design = json.loads((G4 / "design.json").read_text())
    lookup = {c["key"]: c for c in design["cells"]}
    runs = rows(G4 / "analysis/runs.csv")
    summary = json.loads((G4 / "analysis/summary.json").read_text())
    assert summary["audited"] and len(runs) == summary["runs"] == 2640
    by = defaultdict(list)
    for r in runs:
        by[r["cell"], r["arm"]].append(float(r["ttft_ms_mean"]))
    assert len(by) == 132*5 and all(len(v) == 4 for v in by.values())
    output = []
    for family, name, xlabel in [("capacity", "gpu_capacity_nine", "Configured KV blocks"),
                                  ("load", "gpu_sustained_load", "Arrival rate (requests/s)")]:
        subset = [c for c in lookup.values() if c["family"] == family]
        key = "capacity" if family == "capacity" else "arrival_rps"
        # Derive rates from exact input timestamps if not separately recorded.
        def xvalue(c):
            if key in c:
                return c[key]
            req = json.loads((G4 / c["input"]).read_text())
            return round(1000/(req[1]["arrival_ms"]-req[0]["arrival_ms"]), 8)
        xs = sorted({xvalue(c) for c in subset})
        fig, ax = plt.subplots(figsize=(1.63, 1.7), layout="constrained")
        for ref, label, color in [("srp", "SRP", "#ab72a4"), ("fixed256", "Fixed 256", "#c29342")]:
            means = []
            for x in xs:
                selected = [c for c in subset if xvalue(c) == x]
                values = [100*(1-st.fmean(by[c["key"], "arrival"])/st.fmean(by[c["key"], ref])) for c in selected]
                means.append(st.fmean(values))
                output.append(dict(family=family, x=x, reference=ref, cells=len(values), mean_reduction_pct=means[-1]))
            ax.plot(xs, means, "o-", color=color, label=label, ms=2.5, lw=1)
        ax.axhline(0, color=".5", lw=.7, ls="--")
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("TTFT reduction (%)", fontsize=7)
        ax.set_xticks([512, 2048, 4096, 6144] if family == "capacity" else xs)
        if family == "capacity":
            ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=6, loc="best", handlelength=1.0, borderpad=.2)
        ax.grid(alpha=.15)
        save(fig, name)
    (OUT / "gpu_scope_plot_values.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
