"""Paper-sized Evaluation panels from archived measurements; no new runs."""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import json
import math
import statistics
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
G1 = ROOT / "writing/www/experiments/gpu_baselines_4090_20260930_r3"
G2 = ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1"
OUT = ROOT / "writing/www/manuscript/latex/figures/evaluation"
ARCHIVE = ROOT / "writing/www/figures/evaluation_revision_20261006"
COLORS = dict(arrival="#195c9b", srp="#ab72a4", native="#658797",
              sarathi="#c29342", past_future="#238a83",
              fixed256="#c29342", fixed512="#238a83")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def main():
    runs = read_csv(G1 / "analysis/runs.csv")
    cases = json.loads((ARCHIVE / "case_selection.json").read_text())["cases"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 2.8), layout="constrained")
    for j, case in enumerate(cases):
        for arm, name in [("arrival", "SPARE"), ("srp", "SRP")]:
            row = next(r for r in runs if r["cell"] == case["cell"]
                       and r["arm"] == arm and int(r["repeat"]) == case["repeat"])
            path = G1 / row["result_path"]
            data = json.loads(path.read_text())
            events = defaultdict(int)
            for request in data["requests"]:
                events[request["arrival_ms"]] += 1
                events[request["first_ms"]] -= 1
            times, waiting, count = sorted(events), [], 0
            for t in times:
                count += events[t]
                waiting.append(count)
            area = sum((times[i+1]-times[i])*waiting[i]
                       for i in range(len(times)-1))
            assert math.isclose(area, sum(r["ttft_ms"] for r in data["requests"]),
                                rel_tol=1e-10)
            axes[j, 0].step([t/1000 for t in times], waiting, where="post",
                            color=COLORS[arm], label=name, lw=1.1)
            with gzip.open(path.parent.parent / data["rounds_path"], "rt") as f:
                rounds = json.load(f)
            axes[j, 1].step([(r["step_end_ms"]-r["step_ms"])/1000 for r in rounds],
                            [r["free_before"] for r in rounds], where="post",
                            color=COLORS[arm], lw=1.1)
            cumulative, total = [], 0
            for r in rounds:
                total += r["recomputed_tokens"]
                cumulative.append(total/1000)
            axes[j, 2].step([r["step_end_ms"]/1000 for r in rounds], cumulative,
                            where="post", color=COLORS[arm], lw=1.1)
        for k, label in enumerate(["Waiting requests", "Free KV blocks",
                                    "Recomputed\n(k tokens)"]):
            axes[j, k].set_ylabel(label)
            axes[j, k].grid(alpha=.18)
            axes[j, k].set_xlabel("Time (s)")
            axes[j, k].set_title(f"({chr(97+j*3+k)}) " +
                                ("Conversation: improvement" if j == 0
                                 else "Code: regression"))
        axes[j, 0].legend(loc="upper right")
    save(fig, "compact_mechanism")

    aggregate = {r["arm"]: r for r in read_csv(G2 / "analysis/aggregate.csv")}
    arms = ["arrival", "native", "sarathi", "past_future", "srp"]
    names = ["SPARE", "V1", "Sarathi", "Past-Future", "SRP"]
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.65), layout="constrained",
                             sharey=True)
    for ax, metric, title, unit in zip(
            axes, ["output_throughput_tps", "slo_ttft5000_maxitl200_goodput_rps"],
            ["(a) Output throughput", "(b) SLO goodput"], ["Tokens/s", "Requests/s"]):
        vals = [float(aggregate[a][metric]) for a in arms]
        ax.barh(range(len(arms)), vals, color=[COLORS[a] for a in arms], height=.65)
        ax.set_yticks(range(len(arms)), names)
        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    save(fig, "compact_service")

    stats = json.loads((ARCHIVE / "supplement_statistics.json").read_text())
    fig, ax = plt.subplots(figsize=(3.35, 1.65), layout="constrained")
    capacities = [768, 2048, 4096]
    for arm, label, marker in [("srp", "SRP", "o"), ("fixed256", "Fixed 256", "s"),
                                ("fixed512", "Fixed 512", "^")]:
        values = [stats["paired_capacity"][str(c)][arm] for c in capacities]
        ax.plot(capacities, values, marker=marker, color=COLORS[arm],
                label=label, lw=1.1, ms=3)
    ax.axhline(0, color=".5", lw=.7, ls="--")
    ax.set_xticks(capacities)
    ax.set_xlabel("Configured KV blocks")
    ax.set_ylabel("TTFT reduction (%)")
    ax.grid(alpha=.18)
    ax.legend(ncol=3, loc="upper right", columnspacing=.6, handlelength=1.0)
    save(fig, "compact_capacity")


if __name__ == "__main__":
    main()
