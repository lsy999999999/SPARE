"""Run continuous trace samples with 50 calibrated timing repeats each."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from research.www_evaluation_completion_20261003.common import ROOT, read, write, digest
from research.www_simulator_50rep_20261014.samples import SAMPLE_COUNT, extract
from research.www_simulator_50rep_20261014.noise import residuals


OLD = ROOT / "artifacts/www_evaluation_completion_20261003_r1"
OUT = ROOT / "artifacts/www_simulator_50rep_20261014_r1"
REPEATS_PER_SAMPLE = 50
MASTER_SEED = 20261014
FAMILIES = ("core", "sensitivity", "sustained")
METRICS = (
    "ttft_ms_mean", "ttft_ms_p95", "e2e_ms_mean", "request_throughput_rps",
    "output_throughput_tps", "time_weighted_kv_util", "peak_kv_util",
    "preemptions", "recomputed_tokens", "interventions", "elapsed_simulated_ms",
)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value)
    tmp.replace(path)


def prepare() -> dict:
    target = OUT / "design.json"
    if target.exists():
        return read(target)
    old = read(OLD / "design.json")
    tasks = [task for task in old["tasks"] if task["family"] in FAMILIES]
    assert len(tasks) == 2412 and len({t["name"] for t in tasks}) == len(tasks)
    windows = sorted({task["window"] for task in tasks})
    assert len(windows) == 6
    sample_manifest = extract(OUT)
    timing_residuals = residuals(OLD, OUT)
    assert len(sample_manifest["samples"]) == len(windows) * SAMPLE_COUNT
    hashes = {item["path"]: item["sha256"] for item in sample_manifest["samples"]}
    hashes["artifacts/www_evaluation_completion_20261003_r1/design.json"] = digest(OLD / "design.json")
    hashes["artifacts/www_evaluation_completion_20261003_r1/timing_profile.json"] = digest(OLD / "timing_profile.json")
    hashes["artifacts/www_simulator_50rep_20261014_r1/samples.json"] = digest(OUT / "samples.json")
    hashes["artifacts/www_simulator_50rep_20261014_r1/timing_residuals.json"] = digest(OUT / "timing_residuals.json")
    hashes.update(sample_manifest["source_hashes"])
    for path in old["source_hashes"]:
        hashes[path] = digest(ROOT / path)
    for name in ("run.py", "samples.py", "noise.py"):
        script = str((Path(__file__).parent / name).relative_to(ROOT))
        hashes[script] = digest(ROOT / script)
    design = dict(sample_count=SAMPLE_COUNT, repeats_per_sample=REPEATS_PER_SAMPLE,
                  master_seed=MASTER_SEED, tasks=tasks, windows=windows,
                  samples=sample_manifest["samples"], hashes=hashes, profiles=old["profiles"],
                  expected_per_repeat=len(tasks),
                  expected_total=SAMPLE_COUNT * REPEATS_PER_SAMPLE * len(tasks),
                  sampling=sample_manifest["selection"],
                  timing_perturbation=timing_residuals["method"],
                  inference="10 nonoverlapping request segments per source/date window; "
                            "50 timing repetitions per segment. Dates remain six clusters, "
                            "not 60 independent source days or GPU measurements.",
                  excluded="Historical GPU-matched validation is not resampled because that would break "
                           "request-identity matching. It remains in the original batch.",
                  performance_validated=False)
    write(target, design)
    return design


def verify(design: dict) -> None:
    for path, expected in design["hashes"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Frozen input/source changed: {path}")


def shard_paths(sample: int, repeat: int, window: str) -> tuple[Path, Path]:
    stem = OUT / "shards" / f"{window}_s{sample:02d}_r{repeat:02d}"
    return stem.with_suffix(".jsonl"), stem.with_suffix(".json")


def accepted(sample: int, repeat: int, window: str, expected: int) -> bool:
    data, meta = shard_paths(sample, repeat, window)
    if not (data.exists() and meta.exists()):
        return False
    record = read(meta)
    return (record.get("complete") is True and record.get("rows") == expected
            and record.get("sha256") == digest(data))


def sampled_requests(raw: list[dict], template: list[dict]) -> list[dict]:
    assert len(template) in (128, 512)
    return [dict(id=f"{row['id']}:draw{i}", prompt=row["prompt"], output=row["output"],
                 arrival_ms=template[i]["arrival_ms"])
            for i, row in enumerate(raw[:len(template)])]


def run_shard(arg: tuple[int, int, str, list[dict], str]) -> dict:
    sample, repeat, window, tasks, raw_path = arg
    from functools import lru_cache
    from simulator_core.timing import ResourceTiming
    from research.www_evaluation_completion_20261003 import simulate as simulator
    from research.www_evaluation_completion_20261003.simulate import TIMING_PARAMETERS
    from research.www_evaluation_completion_20261003.timing import FittedTiming
    from research.www_simulator_50rep_20261014.noise import PerturbedTiming, StaircaseTiming

    raw = read(ROOT / raw_path)
    residual = read(OUT / "timing_residuals.json")
    fitted = FittedTiming(read(OLD / "timing_profile.json"))
    simulator.digest = lru_cache(maxsize=None)(simulator.digest)
    requests = {}
    for task in tasks:
        path = task["input"]
        if path not in requests:
            requests[path] = sampled_requests(raw, read(ROOT / path))
    rows = []
    start = time.monotonic()
    try:
        for task in tasks:
            options = dict(task["options"])
            arm = options.pop("engine_arm", "arrival")
            profile = task["profile"]
            base = fitted if profile == "gpu_fitted" else ResourceTiming(dict(
                measurement_scope="synthetic-uncalibrated", parameters=TIMING_PARAMETERS))
            if profile == "resource_staircase":
                base = StaircaseTiming(base)
            key = (f"{MASTER_SEED}:{window}:{sample}:{repeat}:{task['case']}:"
                   f"{task['capacity']}:{profile}").encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
            timing = PerturbedTiming(base, residual, seed)
            result = simulator.simulate(requests[task["input"]], arm=arm,
                                        capacity=task["capacity"], profile="resource",
                                        policy_options=options, fitted_timing=timing,
                                        seed=seed, output=None)
            metrics = result["metrics"]
            rows.append(dict(name=task["name"], sample=sample, repeat=repeat, window=window,
                             family=task["family"], case=task["case"],
                             capacity=task["capacity"], profile=task["profile"],
                             arm=task["arm"], trajectory_sha256=result["trajectory_sha256"],
                             **{key: metrics[key] for key in METRICS}))
        data, meta = shard_paths(sample, repeat, window)
        atomic_text(data, "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
        write(meta, dict(complete=True, sample=sample, repeat=repeat, window=window, rows=len(rows),
                         sha256=digest(data), elapsed_s=time.monotonic() - start))
        return dict(sample=sample, repeat=repeat, window=window, complete=True, rows=len(rows))
    except Exception as exc:
        return dict(sample=sample, repeat=repeat, window=window, complete=False, rows=len(rows),
                    failed_task=task["name"], error=repr(exc))


def status() -> dict:
    design = prepare()
    count = Counter(t["window"] for t in design["tasks"])
    accepted_shards = [(s, r, w) for s in range(SAMPLE_COUNT)
                       for r in range(REPEATS_PER_SAMPLE) for w in design["windows"]
                       if accepted(s, r, w, count[w])]
    expected_shards = SAMPLE_COUNT * REPEATS_PER_SAMPLE * len(count)
    result = dict(accepted_shards=len(accepted_shards), expected_shards=expected_shards,
                  accepted_runs=sum(count[w] for _, _, w in accepted_shards),
                  expected_runs=design["expected_total"],
                  complete=len(accepted_shards) == expected_shards)
    print(json.dumps(result, ensure_ascii=False))
    return result


def run(jobs: int) -> None:
    design = prepare()
    verify(design)
    groups = defaultdict(list)
    for task in design["tasks"]:
        groups[task["window"]].append(task)
    samples = {(item["window"], item["sample"]): item["path"]
               for item in design["samples"]}
    pending = [(sample, repeat, window, groups[window], samples[window, sample])
               for sample in range(SAMPLE_COUNT) for repeat in range(REPEATS_PER_SAMPLE)
               for window in design["windows"]
               if not accepted(sample, repeat, window, len(groups[window]))]
    start = time.monotonic()
    failures = []
    write(OUT / "status.json", dict(stage="running", pending_shards=len(pending),
                                    expected_runs=design["expected_total"], jobs=jobs))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_shard, arg): arg[:3] for arg in pending}
        while futures:
            ready, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for future in ready:
                sample, repeat, window = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(sample=sample, repeat=repeat, window=window,
                                  complete=False, error=repr(exc))
                if not result["complete"]:
                    failures.append(result)
                    print("FAILED", result, flush=True)
            progress = status()
            write(OUT / "status.json", dict(stage="running", **progress,
                                            pending_shards=len(futures), failures=failures,
                                            elapsed_s=time.monotonic() - start, jobs=jobs))
    if not failures and status()["complete"]:
        analyze()
    else:
        write(OUT / "status.json", dict(stage="incomplete", failures=failures,
                                        elapsed_s=time.monotonic() - start, **status()))


def analyze() -> None:
    design = prepare()
    progress = status()
    if not progress["complete"]:
        raise RuntimeError("Cannot analyze an incomplete sample-by-repeat matrix")
    sums = defaultdict(lambda: Counter())
    counts = Counter()
    trajectories = defaultdict(set)
    pair_sums = defaultdict(lambda: [0., 0])
    for sample in range(SAMPLE_COUNT):
        for repeat in range(REPEATS_PER_SAMPLE):
            for window in design["windows"]:
                data, _ = shard_paths(sample, repeat, window)
                rows = [json.loads(line) for line in data.read_text().splitlines()]
                by_cell = defaultdict(dict)
                for row in rows:
                    name = row["name"]
                    counts[name] += 1
                    for key in METRICS:
                        sums[name][key] += row[key]
                    if row["arm"] == "spare":
                        trajectories[name, sample].add(row["trajectory_sha256"])
                    by_cell[row["case"], row["capacity"], row["profile"]][row["arm"]] = row
                for (_, _, profile), arms in by_cell.items():
                    spare = arms.get("spare")
                    if spare is None:
                        continue
                    for arm, row in arms.items():
                        if arm == "spare":
                            continue
                        key = (row["family"], profile, arm, window, sample, repeat)
                        pair_sums[key][0] += 100 * (row["ttft_ms_mean"] - spare["ttft_ms_mean"]) / row["ttft_ms_mean"]
                        pair_sums[key][1] += 1
    assert len(counts) == design["expected_per_repeat"]
    assert all(n == SAMPLE_COUNT * REPEATS_PER_SAMPLE for n in counts.values())
    means = []
    for task in design["tasks"]:
        means.append(dict(name=task["name"], family=task["family"], window=task["window"],
                          case=task["case"], capacity=task["capacity"], profile=task["profile"],
                          arm=task["arm"], samples=SAMPLE_COUNT,
                          repeats_per_sample=REPEATS_PER_SAMPLE,
                          **{key: sums[task["name"]][key] / counts[task["name"]] for key in METRICS}))
    with (OUT / "means.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(means[0])); writer.writeheader(); writer.writerows(means)
    samples = defaultdict(list)
    repeats = defaultdict(list)
    for (family, profile, arm, window, sample, repeat), (total, n) in pair_sums.items():
        assert n > 0
        samples[family, profile, arm, window, sample].append(total / n)
        repeats[family, profile, arm].append(total / n)
    grouped = defaultdict(list)
    for (family, profile, arm, window, sample), values in samples.items():
        assert len(values) == REPEATS_PER_SAMPLE
        grouped[family, profile, arm].append(statistics.fmean(values))
    summary = []
    for (family, profile, arm), values in sorted(grouped.items()):
        assert len(values) == len(design["windows"]) * SAMPLE_COUNT
        sorted_values = sorted(values)
        summary.append(dict(family=family, profile=profile, control=arm,
                            source_dates=len(design["windows"]), samples_per_date=SAMPLE_COUNT,
                            repeats_per_sample=REPEATS_PER_SAMPLE,
                            mean_improvement_pct=statistics.fmean(values),
                            sample_p05_pct=sorted_values[2], sample_p95_pct=sorted_values[57],
                            positive_samples=sum(value > 0 for value in values),
                            positive_repeats=sum(value > 0 for value in repeats[family, profile, arm]),
                            total_repeats=len(repeats[family, profile, arm])))
    with (OUT / "paired_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    unique_trajectories = Counter(len(hashes) for hashes in trajectories.values())
    write(OUT / "summary.json", dict(complete=True, samples_per_date=SAMPLE_COUNT,
                                     repeats_per_sample=REPEATS_PER_SAMPLE,
                                     runs=design["expected_total"], tasks=len(counts),
                                     windows=design["windows"], comparisons=summary,
                                     spare_unique_trajectories_per_sample=dict(unique_trajectories),
                                     performance_validated=False))
    lines = ["# 连续轨迹取样 × 每样本 50 次模拟", "",
             f"已完成 {design['expected_total']} 条 CPU 模拟：六个来源日期各 10 份连续轨迹样本，"
             "每份样本的每个实验单元做 50 次校准扰动回放。",
             "各日期内样本互不重叠；同一份样本与重复编号下所有方法、容量和时间模型配对。",
             "六个来源日期仍是跨来源推断单位。样本和重复不是新的独立日期，也不是 GPU 实测。",
             "", "逐单元均值见 `means.csv`；所有原值见 `shards/`。",
             "配对 TTFT 汇总见 `paired_summary.csv`；P05/P95 描述 60 份轨迹样本均值的波动，不是跨日期置信区间。",
             "历史 GPU 留出验证保留原始请求配对，未纳入重采样。", "",
             "| 实验 | 时间模型 | 对照 | 平均 TTFT 改善 | 样本 P05/P95 | 正向样本 |",
             "|---|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['family']} | {row['profile']} | {row['control']} | "
                     f"{row['mean_improvement_pct']:+.2f}% | "
                     f"[{row['sample_p05_pct']:+.2f}%, {row['sample_p95_pct']:+.2f}%] | "
                     f"{row['positive_samples']}/60 |")
    atomic_text(OUT / "REPORT_CN.md", "\n".join(lines) + "\n")
    write(OUT / "status.json", dict(stage="complete", **progress, report=str(OUT / "REPORT_CN.md")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "analyze"))
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({key: value for key, value in prepare().items() if key not in ("tasks", "hashes")},
                         ensure_ascii=False))
    elif args.command == "run":
        run(args.jobs)
    elif args.command == "status":
        status()
    else:
        analyze()


if __name__ == "__main__":
    main()
