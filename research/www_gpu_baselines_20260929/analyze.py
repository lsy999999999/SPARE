"""Audit completed GPU runs and produce paper-facing aggregate tables."""
from __future__ import annotations

import csv
import gzip
import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_gpu_baselines_20260929.common import *

METRICS = ("ttft_ms_mean", "ttft_ms_p95", "ttft_ms_p99", "e2e_ms_mean",
           "e2e_ms_p95", "e2e_ms_p99", "tpot_ms_mean", "max_itl_ms_p99",
           "request_throughput_rps", "output_throughput_tps",
           "slo_ttft5000_maxitl200_attainment", "peak_kv_util",
           "preemptions", "recomputed_tokens", "schedule_ms_mean")


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values, seed):
    rng = random.Random(seed)
    samples = [st.fmean(rng.choice(values) for _ in values) for _ in range(10000)]
    samples.sort()
    return samples[249], samples[9749]


def main():
    design = verify = read(OUT / "design.json")
    manifest = read(OUT / "manifest.json")
    if digest(OUT / "design.json") != manifest["design_sha256"]:
        raise RuntimeError("design/manifest mismatch")
    rows = []
    for cell in design["cells"]:
        group = read(OUT / "groups" / f"{cell['key']}.json")
        if group["state"] != "complete":
            raise RuntimeError(f"incomplete group: {cell['key']}")
        folder = OUT / group["folder"]
        if not read(folder / "aa.json")["passed"]:
            raise RuntimeError(f"A/A failed: {cell['key']}")
        for repeat in range(design["repeats"]):
            for arm in ARMS:
                path = folder / "runs" / f"{arm}_r{repeat}.json"
                result = read(path)
                if result["status"] != "complete" or result["cell"] != cell["key"]:
                    raise RuntimeError(f"invalid run: {path}")
                round_path = folder / result["rounds_path"]
                if digest(round_path) != result["rounds_sha256"]:
                    raise RuntimeError(f"round hash mismatch: {path}")
                with gzip.open(round_path, "rt") as handle:
                    rounds = json.load(handle)
                if len(rounds) != result["metrics"]["rounds"]:
                    raise RuntimeError(f"round count mismatch: {path}")
                row = dict(cell=cell["key"], source_input=cell["source_input"],
                    capacity=cell["capacity"], arm=arm, repeat=repeat,
                    gpu=result["gpu"], order_position=result["order_position"],
                    deployable=result["deployable"],
                    uses_realized_output_lengths=result["uses_realized_output_lengths"],
                    result_path=str(path.relative_to(OUT)), sha256=digest(path))
                row.update({metric: result["metrics"].get(metric) for metric in METRICS})
                rows.append(row)
    if len(rows) != design["expected_formal_runs"]:
        raise RuntimeError((len(rows), design["expected_formal_runs"]))
    aggregate = []
    by_cell_arm = {}
    for row in rows:
        by_cell_arm.setdefault((row["cell"], row["arm"]), []).append(row)
    for arm in ARMS:
        cells = [by_cell_arm[cell["key"], arm] for cell in design["cells"]]
        entry = dict(arm=arm, cells=len(cells), repeats=design["repeats"],
            deployable=arm not in ORACLE_ARMS,
            uses_realized_output_lengths=arm in ORACLE_ARMS)
        for metric in METRICS:
            values = [st.fmean(row[metric] for row in group) for group in cells]
            entry[metric] = st.fmean(values)
        aggregate.append(entry)
    relative = []
    for arm in ARMS:
        if arm == "arrival":
            continue
        improvements = []
        for cell in design["cells"]:
            ours = st.fmean(row["ttft_ms_mean"] for row in by_cell_arm[cell["key"], "arrival"])
            baseline = st.fmean(row["ttft_ms_mean"] for row in by_cell_arm[cell["key"], arm])
            improvements.append(100 * (baseline - ours) / baseline)
        low, high = bootstrap_ci(improvements, int(object_digest(arm)[:12], 16))
        relative.append(dict(reference=arm, cells=len(improvements),
            arrival_mean_ttft_improvement_pct=st.fmean(improvements),
            bootstrap_cell_95ci_low_pct=low, bootstrap_cell_95ci_high_pct=high,
            wins=sum(value > 0 for value in improvements),
            ties=sum(abs(value) < 1e-12 for value in improvements),
            losses=sum(value < 0 for value in improvements)))
    analysis = OUT / "analysis"
    csv_write(analysis / "runs.csv", rows)
    csv_write(analysis / "aggregate.csv", aggregate)
    csv_write(analysis / "relative_to_arrival.csv", relative)
    write(analysis / "summary.json", dict(created=time.time(), audited=True,
        runs=len(rows), groups=len(design["cells"]), aggregate=aggregate,
        relative_to_arrival=relative,
        ci="10,000 deterministic cell-level bootstrap resamples after averaging repeats"))
    lines = ["# WWW baseline RTX 4090 对比结果", "",
        "本报告由冻结的16场景×13方法×4重复矩阵自动生成。GPU执行使用共享vLLM V1后端；oracle/proxy读取真实输出长度，不可部署。论文规则适配不等于各论文原服务系统复现。", "",
        "| 方法 | mean TTFT ms | P99 TTFT ms | mean E2E ms | P99 E2E ms | request/s | SLO达标 | 抢占 | 重算token |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in aggregate:
        lines.append(f"| {row['arm']} | {row['ttft_ms_mean']:.1f} | {row['ttft_ms_p99']:.1f} | {row['e2e_ms_mean']:.1f} | {row['e2e_ms_p99']:.1f} | {row['request_throughput_rps']:.3f} | {row['slo_ttft5000_maxitl200_attainment']:.1%} | {row['preemptions']:.1f} | {row['recomputed_tokens']:.1f} |")
    lines += ["", "## arrival 相对各方法的 mean TTFT", "",
        "正值表示 arrival 更快；置信区间按场景重采样，先在场景内平均4次重复。", "",
        "| 对照 | 平均改善 | 95% CI | 场景胜/平/负 |", "| --- | ---: | ---: | ---: |"]
    for row in relative:
        lines.append(f"| {row['reference']} | {row['arrival_mean_ttft_improvement_pct']:+.2f}% | [{row['bootstrap_cell_95ci_low_pct']:+.2f}%, {row['bootstrap_cell_95ci_high_pct']:+.2f}%] | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines += ["", "环境、模型与GPU UUID见 `manifest.json`，原始逐请求与逐轮证据见各 attempt 目录。", ""]
    (HERE / "RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(audited=True, runs=len(rows), output=str(analysis)), indent=2))


if __name__ == "__main__":
    main()
