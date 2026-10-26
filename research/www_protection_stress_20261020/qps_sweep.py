"""Paired protection check across frozen request motifs and offered QPS."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path

import numpy as np

from research.www_protection_stress_20261020 import run

OUT = run.ROOT / "artifacts/www_protection_stress_20261020_r4"
BACKGROUNDS = (384, 400, 416)
HEAVY_PROMPTS = (1200, 1400, 1600)
HEAVY_COUNTS = (1, 2)
RATES = {
    1: (0.6, 0.8, 1.0, 1.1, 70 / 60, 1.2, 73 / 60, 74 / 60, 1.3,
        1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2),
    2: (0.6, 0.8, 1.0, 1.1, 1.16, 1.2, 1.22, 1.24, 1.3, 1.4, 1.5),
}
ARMS = ("full", "no_finish_guard")
TARGET_COUNTS = (1, 2, 3, 4)
JITTER_RATES = (1.0, 1.2, 1.3, 1.4, 1.5)
JITTER_SEEDS = (11, 23, 47)


def motif(index: int, start_ms: float, background: int, heavy_prompt: int,
          heavy_count: int) -> list[dict]:
    prefix = f"m{index:03d}"
    reqs = [dict(id=f"{prefix}:background_{i}", prompt=background,
                 output=64, arrival_ms=start_ms) for i in range(3)]
    reqs.append(dict(id=f"{prefix}:target", prompt=480, output=16,
                     arrival_ms=start_ms))
    reqs.extend(dict(id=f"{prefix}:heavy_{i}", prompt=heavy_prompt,
                     output=64, arrival_ms=start_ms + run.HEAVY_OFFSET_MS)
                for i in range(heavy_count))
    return reqs


def requests(rate: float | None, background: int, heavy_prompt: int,
             heavy_count: int) -> list[dict]:
    if rate is None:
        return motif(0, 0., background, heavy_prompt, heavy_count)
    size = 4 + heavy_count
    cycles = round(rate * run.HORIZON_MS / (1000 * size))
    actual_rate = cycles * size * 1000 / run.HORIZON_MS
    if not math.isclose(actual_rate, rate, abs_tol=1e-9):
        raise ValueError(f"{rate} QPS cannot use complete {size}-request motifs")
    period = run.HORIZON_MS / cycles
    reqs = [req for index in range(cycles)
            for req in motif(index, (index + .5) * period,
                             background, heavy_prompt, heavy_count)]
    assert reqs[-1]["arrival_ms"] < run.HORIZON_MS
    assert math.isclose(len(reqs) * 1000 / run.HORIZON_MS, rate, abs_tol=1e-9)
    return reqs


def composition_motif(index: int, start_ms: float, background: int,
                      target_count: int) -> list[dict]:
    prefix = f"m{index:03d}"
    reqs = [dict(id=f"{prefix}:background_{i}", prompt=background,
                 output=64, arrival_ms=start_ms) for i in range(4 - target_count)]
    reqs.extend(dict(id=f"{prefix}:target_{i}:target", prompt=480,
                     output=16, arrival_ms=start_ms) for i in range(target_count))
    reqs.extend(dict(id=f"{prefix}:heavy_{i}", prompt=1200,
                     output=64, arrival_ms=start_ms + run.HEAVY_OFFSET_MS)
                for i in range(2))
    return reqs


def composition_requests(rate: float | None, background: int,
                         target_count: int) -> list[dict]:
    if rate is None:
        return composition_motif(0, 0., background, target_count)
    cycles = round(rate * run.HORIZON_MS / 6000)
    if not math.isclose(cycles * 6000 / run.HORIZON_MS, rate, abs_tol=1e-9):
        raise ValueError(f"{rate} QPS cannot use complete six-request motifs")
    period = run.HORIZON_MS / cycles
    reqs = [req for index in range(cycles)
            for req in composition_motif(index, (index + .5) * period,
                                         background, target_count)]
    assert math.isclose(len(reqs) * 1000 / run.HORIZON_MS, rate, abs_tol=1e-9)
    return reqs


def task(args: tuple[int, int, int, float | None, str]) -> dict:
    background, heavy_prompt, heavy_count, rate, arm = args
    reqs = requests(rate, background, heavy_prompt, heavy_count)
    result = run.replay(reqs, arm)
    result.update(background_prompt=background, heavy_prompt=heavy_prompt,
                  heavy_count=heavy_count,
                  offered_qps=len(reqs) * 1000 / run.HORIZON_MS if rate is not None else None)
    return result


def composition_task(args: tuple[int, int, float | None, str]) -> dict:
    background, target_count, rate, arm = args
    reqs = composition_requests(rate, background, target_count)
    result = run.replay(reqs, arm, details=rate is None)
    if rate is None:
        records = result.pop("records")
        result["all_mean_ttft_micro_ms"] = float(np.mean(
            [r["ttft_ms"] for r in records]))
    result.update(background_prompt=background, target_count=target_count,
                  heavy_prompt=1200, heavy_count=2,
                  offered_qps=len(reqs) * 1000 / run.HORIZON_MS if rate is not None else None)
    return result


def jitter_task(args: tuple[int, float, int, str]) -> dict:
    background, rate, seed, arm = args
    cycles = round(rate * run.HORIZON_MS / 6000)
    assert math.isclose(cycles * 6000 / run.HORIZON_MS, rate, abs_tol=1e-9)
    period = run.HORIZON_MS / cycles
    rng = np.random.default_rng(seed)
    starts = [(index + .5) * period + rng.uniform(-.1, .1) * period
              for index in range(cycles)]
    reqs = [req for index, start in enumerate(starts)
            for req in composition_motif(index, start, background, 2)]
    assert reqs[-1]["arrival_ms"] < run.HORIZON_MS
    result = run.replay(reqs, arm)
    result.update(background_prompt=background, target_count=2,
                  heavy_prompt=1200, heavy_count=2, seed=seed,
                  jitter_fraction_of_period=.1,
                  offered_qps=len(reqs) * 1000 / run.HORIZON_MS)
    return result


def tasks(micro: bool) -> list[tuple[int, int, int, float | None, str]]:
    return [(background, heavy_prompt, heavy_count, rate, arm)
            for background in BACKGROUNDS for heavy_prompt in HEAVY_PROMPTS
            for heavy_count in HEAVY_COUNTS
            for rate in ((None,) if micro else RATES[heavy_count]) for arm in ARMS]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def execute(micro: bool, workers: int) -> None:
    design = dict(background_prompts=BACKGROUNDS, heavy_prompts=HEAVY_PROMPTS,
                  heavy_counts=HEAVY_COUNTS, rates_qps=RATES, arms=ARMS,
                  capacity_blocks=run.CAPACITY, heavy_offset_ms=run.HEAVY_OFFSET_MS,
                  horizon_ms=run.HORIZON_MS, warm_ms=run.WARM_MS,
                  motif="three background, one target, one or two heavy requests",
                  qps="exact request count / 300 s", simulator_only=True)
    save(OUT / "design.json", design)
    path = OUT / ("micro.json" if micro else "rates.json")
    rows = json.loads(path.read_text()) if path.exists() else []
    completed = {(r["background_prompt"], r["heavy_prompt"], r["heavy_count"],
                  r["offered_qps"], r["arm"]) for r in rows}
    todo = [item for item in tasks(micro)
            if (item[0], item[1], item[2], item[3], item[4]) not in completed]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(task, item): item for item in todo}
        for index, future in enumerate(as_completed(pending), 1):
            row = future.result()
            rows.append(row)
            print(f"{index}/{len(todo)} background={row['background_prompt']} "
                  f"heavy={row['heavy_prompt']}x{row['heavy_count']} "
                  f"qps={row['offered_qps']} arm={row['arm']}", flush=True)
    rows.sort(key=lambda r: (r["background_prompt"], r["heavy_prompt"],
                             r["heavy_count"], r["offered_qps"] or 0, r["arm"]))
    save(path, rows)


def execute_composition(micro: bool, workers: int) -> None:
    save(OUT / "composition_design.json",
         dict(background_prompts=BACKGROUNDS, target_counts=TARGET_COUNTS,
              target_prompt=480, target_output=16, initial_requests=4,
              heavy_prompt=1200, heavy_output=64, heavy_count=2,
              heavy_offset_ms=run.HEAVY_OFFSET_MS, capacity_blocks=run.CAPACITY,
              rates_qps=RATES[2], horizon_ms=run.HORIZON_MS,
              warm_ms=run.WARM_MS, simulator_only=True))
    path = OUT / ("composition_micro.json" if micro else "composition_rates.json")
    rows = json.loads(path.read_text()) if path.exists() else []
    completed = {(r["background_prompt"], r["target_count"], r["offered_qps"],
                  r["arm"]) for r in rows}
    task_rates = (None,) if micro else RATES[2]
    todo = [(background, target_count, rate, arm)
            for background in BACKGROUNDS for target_count in TARGET_COUNTS
            for rate in task_rates for arm in ARMS
            if (background, target_count, rate, arm) not in completed]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(composition_task, item): item for item in todo}
        for index, future in enumerate(as_completed(pending), 1):
            row = future.result()
            rows.append(row)
            print(f"{index}/{len(todo)} background={row['background_prompt']} "
                  f"targets={row['target_count']} qps={row['offered_qps']} "
                  f"arm={row['arm']}", flush=True)
    rows.sort(key=lambda r: (r["background_prompt"], r["target_count"],
                             r["offered_qps"] or 0, r["arm"]))
    save(path, rows)


def execute_jitter(workers: int) -> None:
    save(OUT / "jitter_design.json",
         dict(background_prompts=BACKGROUNDS, target_count=2,
              heavy_prompt=1200, heavy_count=2, rates_qps=JITTER_RATES,
              seeds=JITTER_SEEDS, jitter_fraction_of_period=.1,
              horizon_ms=run.HORIZON_MS, warm_ms=run.WARM_MS,
              capacity_blocks=run.CAPACITY, simulator_only=True))
    path = OUT / "jitter_rates.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    completed = {(r["background_prompt"], r["offered_qps"], r["seed"],
                  r["arm"]) for r in rows}
    todo = [(background, rate, seed, arm) for background in BACKGROUNDS
            for rate in JITTER_RATES for seed in JITTER_SEEDS for arm in ARMS
            if (background, rate, seed, arm) not in completed]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(jitter_task, item): item for item in todo}
        for index, future in enumerate(as_completed(pending), 1):
            row = future.result()
            rows.append(row)
            print(f"{index}/{len(todo)} background={row['background_prompt']} "
                  f"qps={row['offered_qps']} seed={row['seed']} arm={row['arm']}",
                  flush=True)
    rows.sort(key=lambda r: (r["background_prompt"], r["offered_qps"],
                             r["seed"], r["arm"]))
    save(path, rows)


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((OUT / "rates.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    colors = {"full": "#087E83", "no_finish_guard": "#B64B38"}
    labels = {"full": "Full SPARE", "no_finish_guard": "Without finish guard"}
    for heavy_count in HEAVY_COUNTS:
        fig, axes = plt.subplots(3, 3, figsize=(10.2, 7.2), sharex=True)
        for i, background in enumerate(BACKGROUNDS):
            for j, heavy_prompt in enumerate(HEAVY_PROMPTS):
                ax = axes[i, j]
                for arm in ARMS:
                    points = [r for r in rows if r["background_prompt"] == background
                              and r["heavy_prompt"] == heavy_prompt
                              and r["heavy_count"] == heavy_count and r["arm"] == arm]
                    points.sort(key=lambda r: r["offered_qps"])
                    ax.plot([r["offered_qps"] for r in points],
                            [r["mean_ttft_ms"] / 1000 for r in points],
                            marker="o", markersize=3, linewidth=1.5,
                            color=colors[arm], label=labels[arm])
                ax.axhline(2, color="#888888", linestyle=":", linewidth=.8)
                ax.set_title(f"Background {background}; heavy {heavy_prompt}", fontsize=9)
                ax.grid(axis="y", color="#E3E3E3", linewidth=.5)
                if j == 0:
                    ax.set_ylabel("Mean TTFT (s)")
                if i == 2:
                    ax.set_xlabel("Offered QPS")
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=2,
                   frameon=False, bbox_to_anchor=(.5, 1.01))
        fig.suptitle(f"Finish guard under KV pressure: {heavy_count} heavy request(s)", y=1.045)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"mean_ttft_qps_heavy{heavy_count}.{ext}",
                        bbox_inches="tight", dpi=180)
        plt.close(fig)


def plot_composition() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    periodic = json.loads((OUT / "composition_rates.json").read_text())
    jitter = json.loads((OUT / "jitter_rates.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    colors = {"full": "#087E83", "no_finish_guard": "#B64B38"}
    labels = {"full": "Full SPARE", "no_finish_guard": "Without finish guard"}
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.7), sharex="col", sharey=True)
    for col, background in enumerate(BACKGROUNDS):
        for arm in ARMS:
            points = [r for r in periodic if r["background_prompt"] == background
                      and r["target_count"] == 2 and r["arm"] == arm]
            points.sort(key=lambda r: r["offered_qps"])
            axes[0, col].plot([r["offered_qps"] for r in points],
                              [r["mean_ttft_ms"] / 1000 for r in points],
                              marker="o", markersize=3, linewidth=1.4,
                              color=colors[arm], label=labels[arm])
            rates = []
            means = []
            lower = []
            upper = []
            for rate in JITTER_RATES:
                values = [r["mean_ttft_ms"] / 1000 for r in jitter
                          if r["background_prompt"] == background
                          and r["offered_qps"] == rate and r["arm"] == arm]
                mean = float(np.mean(values))
                rates.append(rate + (-.006 if arm == "full" else .006))
                means.append(mean)
                lower.append(mean - min(values))
                upper.append(max(values) - mean)
            axes[1, col].errorbar(rates, means, yerr=[lower, upper],
                                  marker="o", markersize=3, linewidth=1.4,
                                  capsize=2, color=colors[arm])
        for row in range(2):
            axes[row, col].axhline(2, color="#777777", linestyle=":", linewidth=.8)
            axes[row, col].grid(axis="y", color="#E4E4E4", linewidth=.5)
            axes[row, col].set_ylim(0, 6.6)
        axes[0, col].set_title(f"Background prompt {background}", fontsize=9)
        axes[1, col].set_xlabel("Offered QPS")
    axes[0, 0].set_ylabel("Periodic\nMean TTFT (s)")
    axes[1, 0].set_ylabel("Jittered (3 seeds)\nMean TTFT (s)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(.5, 1.04))
    fig.suptitle("Two near-finished prefill requests under KV pressure", y=1.09)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"composition_mean_ttft_qps.{ext}",
                    bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-composition", action="store_true")
    parser.add_argument("--composition-micro", action="store_true")
    parser.add_argument("--composition-scan", action="store_true")
    parser.add_argument("--jitter", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.micro:
        execute(True, args.workers)
    if args.scan:
        execute(False, args.workers)
    if args.plot:
        plot()
    if args.plot_composition:
        plot_composition()
    if args.composition_micro:
        execute_composition(True, args.workers)
    if args.composition_scan:
        execute_composition(False, args.workers)
    if args.jitter:
        execute_jitter(args.workers)
