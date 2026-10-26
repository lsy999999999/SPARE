"""Replay SPARE and external paper adapters on 50 perturbed draws per trace sample."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from pathlib import Path

from research.www_evaluation_completion_20261003.common import ROOT, digest, read, write
from research.www_simulator_50rep_20261014 import run as trace_run
from research.www_pareto_papers_20261014 import run as papers


OUT = ROOT / "artifacts/www_paper_baselines_50rep_20261015_r1"
OLD = ROOT / "artifacts/www_evaluation_completion_20261003_r1"
PARENT = trace_run.OUT
REPEATS = 50
SAMPLES = 10
BUDGET = 1024
CAPACITIES = (768, 2048, 4096)
MAIN = papers.MAIN
DIAGNOSTIC = papers.DIAGNOSTIC
ARMS = MAIN + DIAGNOSTIC
METRICS = (
    "ttft_ms_mean", "ttft_ms_p95", "e2e_ms_mean", "e2e_ms_p95",
    "request_throughput_rps", "output_throughput_tps", "time_weighted_kv_util",
    "peak_kv_util", "preemptions", "recomputed_tokens", "interventions",
    "elapsed_simulated_ms", "cpu_schedule_including_audit_ms_mean",
)


def history_before_anchors() -> dict[str, list[int]]:
    from research.www_simulator_50rep_20261014.samples import ANCHORS

    result = {}
    for source, anchors in ANCHORS.items():
        cutoff = min(anchors)
        path = ROOT / f"data/azure_{'conv' if source == 'conversation' else 'code'}_1week.csv"
        values = deque(maxlen=1000)
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for row in reader:
                if row[0][:16] >= cutoff:
                    break
                prompt, output = int(row[1]), int(row[2])
                if prompt > 0 and 0 < output <= 512 and prompt + output <= 8000:
                    values.append(output)
        if len(values) != 1000:
            raise RuntimeError(f"Insufficient pre-anchor history for {source}: {len(values)}")
        result[source] = list(values)
    return result


def prepare() -> dict:
    target = OUT / "design.json"
    if target.exists():
        return read(target)
    parent = trace_run.prepare()
    old = read(OLD / "design.json")
    selected = [task for task in old["tasks"]
                if task["arm"] == "spare" and task["profile"] == "gpu_fitted"
                and task["family"] in ("core", "sustained")]
    by_case = defaultdict(dict)
    for task in selected:
        key = (task["window"], task["case"])
        by_case[key][task["capacity"]] = task
    cases = []
    for (window, case), variants in sorted(by_case.items()):
        if set(variants) != set(CAPACITIES):
            raise RuntimeError(f"Incomplete capacities for {case}: {set(variants)}")
        representative = variants[CAPACITIES[0]]
        if any(task["input"] != representative["input"] for task in variants.values()):
            raise RuntimeError(f"Mismatched arrival template for {case}")
        cases.append(dict(window=window, name=case, family=representative["family"],
                          input=representative["input"], requests=128 if representative["family"] == "core" else 512))
    if len(cases) != 36:
        raise RuntimeError(f"Expected 36 arrival cases, found {len(cases)}")
    write(OUT / "history.json", history_before_anchors())
    for path, expected in parent["hashes"].items():
        if not path.endswith(".py") and digest(ROOT / path) != expected:
            raise RuntimeError(f"Frozen input changed since parent batch: {path}")
    hashes = {path: digest(ROOT / path) for path in parent["hashes"]}
    additions = [
        "research/www_paper_baselines_50rep_20261015/run.py",
        "research/www_pareto_papers_20261014/run.py",
        "research/www_legacy_baselines_20261006/scheduler.py",
        "research/www_legacy_baselines_20261006/policy.py",
        "research/www_baseline_fidelity_20261013/scheduler.py",
        "research/www_baseline_fidelity_20261013/policy.py",
        "research/www_evaluation_gpu_completion_20261003/scheduler.py",
        "research/www_evaluation_gpu_completion_20261003/policy.py",
        "artifacts/www_baseline_fidelity_20261013_r1/calibration/cost_model.json",
        "artifacts/www_paper_baselines_50rep_20261015_r1/history.json",
    ]
    hashes.update({path: digest(ROOT / path) for path in additions})
    cases_per_window = Counter(case["window"] for case in cases)
    if set(cases_per_window.values()) != {6}:
        raise RuntimeError(f"Incomplete windows: {cases_per_window}")
    design = dict(parent_design_sha256=digest(PARENT / "design.json"),
                  sample_count=SAMPLES, repeats_per_sample=REPEATS,
                  master_seed=20261015, budget=BUDGET, chunk=256, concurrency=32,
                  capacities=CAPACITIES, cases=cases, arms=ARMS,
                  main_arms=MAIN, diagnostic_arms=DIAGNOSTIC,
                  adapter_scopes={arm: papers.SCOPES[arm] for arm in ARMS},
                  samples=parent["samples"], hashes=hashes,
                  expected_shards=6 * SAMPLES * REPEATS * len(cases_per_window),
                  expected_runs=len(cases) * SAMPLES * REPEATS * len(CAPACITIES) * len(ARMS),
                  history="Last 1000 eligible output lengths strictly before each source's first anchor; disjoint from evaluation samples.",
                  arrival="Original contiguous request order and lengths; previous steady/burst-stop/rps arrival templates.",
                  perturbation=parent["timing_perturbation"],
                  inference="Six source dates; 10 nonoverlapping windows/date and 50 timing draws/window are paired repeated measurements, not independent dates.",
                  performance_validated=False)
    write(target, design)
    return design


def verify(design: dict, host: str) -> None:
    for path, expected in design["hashes"].items():
        if host == "remote" and path.startswith("data/azure_"):
            continue
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Frozen file changed: {path}")
    if digest(PARENT / "design.json") != design["parent_design_sha256"]:
        raise RuntimeError("Parent sample design changed")


def shard_paths(window: str, sample: int, repeat: int, case: str) -> tuple[Path, Path]:
    stem = OUT / "shards" / f"{case}_s{sample:02d}_r{repeat:02d}"
    return stem.with_suffix(".jsonl"), stem.with_suffix(".json")


def accepted(window: str, sample: int, repeat: int, case: str) -> bool:
    data, meta = shard_paths(window, sample, repeat, case)
    if not data.exists() or not meta.exists():
        return False
    record = read(meta)
    return (record.get("complete") is True and record.get("case") == case
            and record.get("rows") == len(ARMS) * len(CAPACITIES)
            and record.get("sha256") == digest(data))


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(content)
    temp.replace(path)


def initialize_worker() -> None:
    global BASE, RESIDUAL, HISTORY
    from research.www_evaluation_completion_20261003 import simulate as simulator
    from research.www_evaluation_completion_20261003.timing import FittedTiming
    from research.www_legacy_baselines_20261006.scheduler import GPUBaselineScheduler as Legacy
    from research.www_baseline_fidelity_20261013.scheduler import GPUBaselineScheduler as Enhanced

    simulator.digest = lru_cache(maxsize=None)(simulator.digest)
    papers.SIM = simulator
    papers.ORIGINAL_MAKE = simulator.make_scheduler
    papers.LEGACY_CLASS = Legacy
    papers.ENHANCED_CLASS = Enhanced
    papers.COST = read(ROOT / "artifacts/www_baseline_fidelity_20261013_r1/calibration/cost_model.json")
    BASE = FittedTiming(read(OLD / "timing_profile.json"))
    RESIDUAL = read(PARENT / "timing_residuals.json")
    HISTORY = read(OUT / "history.json")


def run_shard(task: tuple[dict, int, int, str]) -> dict:
    from research.www_simulator_50rep_20261014.noise import PerturbedTiming

    case, sample, repeat, raw_path = task
    window, name = case["window"], case["name"]
    raw = read(ROOT / raw_path)
    template = read(ROOT / case["input"])
    requests = trace_run.sampled_requests(raw, template)
    source = window.split("_", 1)[0]
    rows = []
    start = time.monotonic()
    failed_arm = None
    try:
        for capacity in CAPACITIES:
            key = f"20261015:{window}:{sample}:{repeat}:{name}:{capacity}".encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
            for arm in ARMS:
                failed_arm = arm
                papers.TIMING = PerturbedTiming(BASE, RESIDUAL, seed)
                result = papers.simulate(requests, arm, capacity, BUDGET,
                                         HISTORY[source], seed)
                metrics = result["metrics"]
                if metrics["completed"] != len(requests):
                    raise RuntimeError(f"Incomplete requests: {metrics['completed']}")
                expected_tokens = sum(r["prompt"] + r["output"] - 1 for r in requests)
                if metrics["executed_tokens"] - metrics["recomputed_tokens"] != expected_tokens:
                    raise RuntimeError("Useful-token conservation failed")
                expected_tps = sum(r["output"] for r in requests) * 1000 / metrics["elapsed_simulated_ms"]
                if abs(metrics["output_throughput_tps"] - expected_tps) > 1e-8:
                    raise RuntimeError("Throughput denominator failed")
                rows.append(dict(window=window, case=name, family=case["family"],
                                 sample=sample, repeat=repeat, capacity=capacity,
                                 arm=arm, diagnostic=arm in DIAGNOSTIC,
                                 trajectory_sha256=result["trajectory_sha256"],
                                 **{metric: metrics[metric] for metric in METRICS}))
        data, meta = shard_paths(window, sample, repeat, name)
        atomic_text(data, "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
        write(meta, dict(complete=True, window=window, case=name,
                         sample=sample, repeat=repeat, rows=len(rows),
                         sha256=digest(data), elapsed_s=time.monotonic() - start))
        return dict(complete=True, case=name, sample=sample, repeat=repeat,
                    rows=len(rows))
    except Exception as exc:
        return dict(complete=False, case=name, sample=sample, repeat=repeat,
                    failed_arm=failed_arm, rows=len(rows), error=repr(exc))


def selected_tasks(design: dict, sample_start: int, sample_stop: int) -> list[tuple]:
    samples = {(item["window"], item["sample"]): item["path"] for item in design["samples"]}
    return [(case, sample, repeat, samples[case["window"], sample])
            for sample in range(sample_start, sample_stop)
            for repeat in range(REPEATS)
            for case in design["cases"]
            if not accepted(case["window"], sample, repeat, case["name"])]


def run(sample_start: int, sample_stop: int, jobs: int, host: str) -> None:
    design = prepare()
    verify(design, host)
    if not (0 <= sample_start < sample_stop <= SAMPLES and jobs > 0):
        raise ValueError("Invalid sample range or worker count")
    tasks = selected_tasks(design, sample_start, sample_stop)
    expected = (sample_stop - sample_start) * REPEATS * len(design["cases"])
    finished = expected - len(tasks)
    failures = []
    start = time.monotonic()
    status_path = OUT / f"partition_status_{host}.json"

    def publish(stage: str, pending: int) -> None:
        write(status_path, dict(stage=stage, host=host, sample_start=sample_start,
                                sample_stop=sample_stop, jobs=jobs,
                                accepted_shards=finished, expected_shards=expected,
                                accepted_runs=finished * len(ARMS) * len(CAPACITIES),
                                remaining=pending, failures=failures,
                                elapsed_s=time.monotonic() - start))

    publish("running", len(tasks))
    with ProcessPoolExecutor(max_workers=jobs, initializer=initialize_worker) as pool:
        futures = {pool.submit(run_shard, task): task[:3] for task in tasks}
        while futures:
            ready, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for future in ready:
                case, sample, repeat = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(complete=False, case=case["name"], sample=sample,
                                  repeat=repeat, error=repr(exc))
                if result["complete"] and accepted(case["window"], sample, repeat, case["name"]):
                    finished += 1
                else:
                    failures.append(result)
                    print("FAILED", result, flush=True)
            publish("running", len(futures))
    publish("complete" if not failures and finished == expected else "incomplete", 0)
    print(host, finished, "/", expected, "shards; failures", len(failures), flush=True)


def status() -> dict:
    design = prepare()
    done = sum(accepted(case["window"], sample, repeat, case["name"])
               for case in design["cases"] for sample in range(SAMPLES)
               for repeat in range(REPEATS))
    result = dict(accepted_shards=done, expected_shards=design["expected_shards"],
                  accepted_runs=done * len(ARMS) * len(CAPACITIES),
                  expected_runs=design["expected_runs"],
                  complete=done == design["expected_shards"])
    print(json.dumps(result))
    return result


def analyze() -> None:
    design = prepare()
    progress = status()
    if not progress["complete"]:
        raise RuntimeError("Cannot analyze incomplete paper-baseline matrix")
    sums = defaultdict(lambda: Counter())
    counts = Counter()
    sample_pairs = defaultdict(lambda: [0.0, 0.0, 0])
    trajectories = defaultdict(set)
    for case in design["cases"]:
        for sample in range(SAMPLES):
            for repeat in range(REPEATS):
                data, _ = shard_paths(case["window"], sample, repeat, case["name"])
                by_capacity = defaultdict(dict)
                for line in data.read_text().splitlines():
                    row = json.loads(line)
                    key = (row["family"], row["window"], row["case"], row["capacity"], row["arm"])
                    counts[key] += 1
                    for metric in METRICS:
                        sums[key][metric] += row[metric]
                    by_capacity[row["capacity"]][row["arm"]] = row
                    if row["arm"] == "arrival":
                        trajectories[row["case"], row["capacity"], sample].add(row["trajectory_sha256"])
                for capacity, arms in by_capacity.items():
                    spare = arms["arrival"]
                    for arm in ARMS:
                        if arm == "arrival":
                            continue
                        baseline = arms[arm]
                        key = (case["family"], arm, case["window"], sample, repeat)
                        sample_pairs[key][0] += 100 * (baseline["ttft_ms_mean"] - spare["ttft_ms_mean"]) / baseline["ttft_ms_mean"]
                        sample_pairs[key][1] += 100 * (spare["output_throughput_tps"] - baseline["output_throughput_tps"]) / baseline["output_throughput_tps"]
                        sample_pairs[key][2] += 1
    if len(counts) != len(design["cases"]) * len(CAPACITIES) * len(ARMS):
        raise RuntimeError("Missing experiment cells")
    if set(counts.values()) != {SAMPLES * REPEATS}:
        raise RuntimeError("Uneven repeat counts")
    means = []
    for key, count in sorted(counts.items()):
        family, window, case, capacity, arm = key
        means.append(dict(family=family, window=window, case=case,
                          capacity=capacity, arm=arm, diagnostic=arm in DIAGNOSTIC,
                          samples=SAMPLES, repeats_per_sample=REPEATS,
                          **{metric: sums[key][metric] / count for metric in METRICS}))
    with (OUT / "means.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(means[0]))
        writer.writeheader(); writer.writerows(means)
    by_sample = defaultdict(list)
    for (family, arm, window, sample, repeat), (ttft, throughput, n) in sample_pairs.items():
        expected_n = (2 if family == "core" else 4) * len(CAPACITIES)
        if n != expected_n:
            raise RuntimeError(f"Incomplete pair: {family}, {arm}, {window}, {sample}, {repeat}: {n}")
        by_sample[family, arm, window, sample].append((ttft / n, throughput / n))
    by_arm = defaultdict(list)
    for (family, arm, window, sample), values in by_sample.items():
        if len(values) != REPEATS:
            raise RuntimeError("Incomplete sample repeats")
        by_arm[family, arm].append((statistics.fmean(v[0] for v in values),
                                    statistics.fmean(v[1] for v in values)))
    summary = []
    for (family, arm), values in sorted(by_arm.items()):
        if len(values) != 6 * SAMPLES:
            raise RuntimeError("Incomplete source samples")
        ordered_ttft = sorted(v[0] for v in values)
        ordered_throughput = sorted(v[1] for v in values)
        summary.append(dict(family=family, arm=arm, diagnostic=arm in DIAGNOSTIC,
                            source_dates=6, samples_per_date=SAMPLES,
                            repeats_per_sample=REPEATS,
                            ttft_improvement_pct=statistics.fmean(ordered_ttft),
                            ttft_sample_p05_pct=ordered_ttft[2],
                            ttft_sample_p95_pct=ordered_ttft[57],
                            ttft_positive_samples=sum(v > 0 for v in ordered_ttft),
                            throughput_improvement_pct=statistics.fmean(ordered_throughput),
                            throughput_sample_p05_pct=ordered_throughput[2],
                            throughput_sample_p95_pct=ordered_throughput[57],
                            throughput_positive_samples=sum(v > 0 for v in ordered_throughput)))
    with (OUT / "paired_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    unique = Counter(len(v) for v in trajectories.values())
    write(OUT / "summary.json", dict(**progress, main_arms=MAIN,
                                     diagnostic_arms=DIAGNOSTIC,
                                     comparisons=summary,
                                     spare_unique_trajectories_per_sample=dict(unique),
                                     performance_validated=False))
    lines = ["# SPARE 与外部论文方法：连续轨迹 × 50 次耗时扰动", "",
             f"已验收 {progress['accepted_runs']} 条 CPU 模拟，"
             "六个来源日期各 10 段连续轨迹，每段每实验单元 50 次配对回放。",
             "主表仅列可见信息规则/代理；真值长度 oracle 另列诊断。",
             "正值表示 SPARE 的 mean TTFT 更低或 output tokens/s 更高。",
             "各日期内窗口及扰动不是独立来源；模拟不代表完整论文系统的 GPU 实测。", "",
             "| 实验族 | 方法 | TTFT 改善 | TTFT 正向样本 | 吞吐改善 | 吞吐正向样本 |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        if row["diagnostic"]:
            continue
        lines.append(f"| {row['family']} | {papers.LABELS[row['arm']]} | "
                     f"{row['ttft_improvement_pct']:+.2f}% | {row['ttft_positive_samples']}/60 | "
                     f"{row['throughput_improvement_pct']:+.2f}% | {row['throughput_positive_samples']}/60 |")
    lines.extend(["", "## 真值长度诊断", "",
                  "下列实现读取真实输出长度，不能纳入可部署方法排名。", "",
                  "| 实验族 | 方法 | TTFT 改善 | TTFT 正向样本 | 吞吐改善 | 吞吐正向样本 |",
                  "|---|---|---:|---:|---:|---:|"])
    for row in summary:
        if not row["diagnostic"]:
            continue
        lines.append(f"| {row['family']} | {papers.LABELS[row['arm']]} | "
                     f"{row['ttft_improvement_pct']:+.2f}% | {row['ttft_positive_samples']}/60 | "
                     f"{row['throughput_improvement_pct']:+.2f}% | {row['throughput_positive_samples']}/60 |")
    atomic_text(OUT / "REPORT_CN.md", "\n".join(lines) + "\n")
    write(OUT / "status.json", dict(stage="complete", **progress,
                                    report=str(OUT / "REPORT_CN.md")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "analyze"))
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-stop", type=int, default=SAMPLES)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--host", choices=("local", "remote"), default="local")
    args = parser.parse_args()
    if args.command == "prepare":
        design = prepare()
        print(json.dumps({key: value for key, value in design.items()
                          if key not in ("cases", "samples", "hashes", "adapter_scopes")}))
    elif args.command == "run":
        run(args.sample_start, args.sample_stop, args.jobs, args.host)
    elif args.command == "status":
        status()
    else:
        analyze()


if __name__ == "__main__":
    main()
