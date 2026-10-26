"""Audit held-out fixed-control GPU runs and paired window uncertainty."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_evaluation_gpu_completion_20261003.common import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_plan_abc_20261008_r1/c"
METRICS = ("ttft_ms_mean", "ttft_ms_p99", "e2e_ms_mean", "e2e_ms_p99",
           "tpot_ms_mean", "max_itl_ms_p99", "request_throughput_rps",
           "slo_ttft5000_maxitl200_attainment", "preemptions", "recomputed_tokens")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci(values, seed):
    rng = random.Random(seed)
    values = list(values)
    samples = sorted(st.fmean(rng.choice(values) for _ in values) for _ in range(10000))
    return [samples[249], samples[9749]]


def audit_run(folder, name, cell, arm, repeat):
    path = folder / "runs" / f"{name}.json"
    result = read(path)
    assert result["status"] == "complete" and result["cell"] == cell["key"]
    assert result["arm"] == arm and result["repeat"] == repeat
    assert result["metrics"]["completed"] == cell["requests"]
    round_path = folder / result["rounds_path"]
    assert digest(round_path) == result["rounds_sha256"], round_path
    return result, digest(path)


def audit_aa(folder, cell, design):
    aa = read(folder / "aa.json")
    assert aa["passed"] and aa["threshold"] == design["aa_threshold"]
    runs = [audit_run(folder, f"aa_srp_r{i}", cell, "srp", i)[0]
            for i in range(design["aa_repeats"])]
    for metric in ("ttft_ms_mean", "elapsed_ms"):
        values = [run["metrics"][metric] for run in runs]
        spread = (max(values) - min(values)) / st.fmean(values)
        assert spread <= design["aa_threshold"]
        assert math.isclose(spread, aa["relative_spread"][metric], rel_tol=1e-9, abs_tol=1e-12)


def audit_calibration(design, selection):
    cells = [cell for cell in design["cells"] if cell["stage"] == "cal"]
    assert len(cells) == 8
    assert selection["test_cells"] == [cell["key"] for cell in design["cells"]
                                       if cell["stage"] == "test"]
    values = defaultdict(list)
    for cell in cells:
        group = read(OUT / "groups" / (cell["key"] + ".json"))
        assert group["state"] == "complete" and group["stage"] == "cal"
        folder = OUT / group["folder"]
        audit_aa(folder, cell, design)
        for arm in design["cal_methods"]:
            runs = [audit_run(folder, f"{arm}_r{i}", cell, arm, i)[0]
                    for i in range(design["cal_repeats"])]
            assert all(run["recorded_at"] < selection["created"] for run in runs)
            values[arm].append(st.fmean(run["metrics"]["ttft_ms_mean"] for run in runs))
    score = {arm: st.fmean(rows) for arm, rows in values.items()}
    assert set(score) == set(selection["scores"])
    for arm, value in score.items():
        assert math.isclose(value, selection["scores"][arm], rel_tol=1e-9, abs_tol=1e-8)
    for family in ("fixed", "guarded"):
        choice = min(design["calibration_budgets"],
                     key=lambda budget: (score[f"{family}_{budget}"], budget))
        assert selection[family] == choice
    return score


def main():
    design = read(OUT / "design.json")
    manifest = read(OUT / "manifest.json")
    assert digest(OUT / "design.json") == manifest["design_sha256"]
    selection = read(OUT / "selection.json")
    assert selection["passed"]
    cal_status = read(OUT / "cal_status.json")
    test_status = read(OUT / "test_status.json")
    assert cal_status["passed"] and cal_status["accepted"] == 8
    assert test_status["passed"] and test_status["accepted"] == 24
    cal_score = audit_calibration(design, selection)
    cells = [cell for cell in design["cells"] if cell["stage"] == "test"]
    assert len(cells) == 24
    runs = []
    cell_means = []
    for cell in cells:
        group = read(OUT / "groups" / (cell["key"] + ".json"))
        assert group["state"] == "complete" and group["stage"] == "test"
        folder = OUT / group["folder"]
        audit_aa(folder, cell, design)
        for arm in design["test_methods"]:
            arm_runs = []
            for repeat in range(design["test_repeats"]):
                path = folder / "runs" / f"{arm}_r{repeat}.json"
                result, result_hash = audit_run(folder, f"{arm}_r{repeat}", cell, arm, repeat)
                assert result["recorded_at"] > selection["created"]
                row = dict(cell=cell["key"], window=cell["window"], source=cell["source"],
                    date=cell["date"], shape=cell["shape"], capacity=cell["capacity"],
                    arm=arm, repeat=repeat, gpu=result["gpu"],
                    order_position=result["order_position"],
                    result_sha256=result_hash, rounds_sha256=result["rounds_sha256"])
                row.update({metric: result["metrics"][metric] for metric in METRICS})
                runs.append(row)
                arm_runs.append(row)
            mean = dict(cell=cell["key"], window=cell["window"], source=cell["source"],
                date=cell["date"], shape=cell["shape"], capacity=cell["capacity"], arm=arm)
            mean.update({metric: st.fmean(row[metric] for row in arm_runs) for metric in METRICS})
            cell_means.append(mean)
    assert len(runs) == 480 and len(cell_means) == 120
    indexed = {(r["cell"], r["arm"]): r for r in cell_means}
    aggregate = []
    for arm in design["test_methods"]:
        group = [r for r in cell_means if r["arm"] == arm]
        mean = dict(arm=arm, cells=len(group))
        mean.update({metric: st.fmean(row[metric] for row in group) for metric in METRICS})
        aggregate.append(mean)
    relative = []
    for arm in design["test_methods"]:
        if arm == "spare":
            continue
        effects = {"ttft": [], "e2e": [], "tail": []}
        by_window = {name: defaultdict(list) for name in effects}
        for cell in cells:
            spare = indexed[cell["key"], "spare"]
            base = indexed[cell["key"], arm]
            for name, metric in (("ttft", "ttft_ms_mean"), ("e2e", "e2e_ms_mean"),
                                 ("tail", "max_itl_ms_p99")):
                effect = 100 * (base[metric] - spare[metric]) / base[metric]
                effects[name].append(effect)
                by_window[name][cell["window"]].append(effect)
        window_means = {name: [st.fmean(group) for group in windows.values()]
                        for name, windows in by_window.items()}
        seed = int(digest(OUT / "design.json")[:8], 16) + len(relative)
        relative.append(dict(reference=arm, mean_improvement_pct=st.fmean(effects["ttft"]),
            window_equal_improvement_pct=st.fmean(window_means["ttft"]),
            window_bootstrap_95ci_pct=ci(window_means["ttft"], seed),
            mean_e2e_improvement_pct=st.fmean(effects["e2e"]),
            window_bootstrap_95ci_e2e_pct=ci(window_means["e2e"], seed),
            mean_tail_improvement_pct=st.fmean(effects["tail"]),
            window_bootstrap_95ci_tail_pct=ci(window_means["tail"], seed),
            wins=sum(x > 0 for x in effects["ttft"]),
            losses=sum(x < 0 for x in effects["ttft"]),
            test_cells=len(effects["ttft"]), independent_windows=len(by_window["ttft"])))
    analysis = OUT / "analysis"
    csv_write(analysis / "runs.csv", runs)
    csv_write(analysis / "cell_means.csv", cell_means)
    csv_write(analysis / "aggregate.csv", aggregate)
    csv_write(analysis / "relative.csv", [dict(row,
        window_bootstrap_95ci_low_pct=row["window_bootstrap_95ci_pct"][0],
        window_bootstrap_95ci_high_pct=row["window_bootstrap_95ci_pct"][1],
        window_bootstrap_95ci_e2e_low_pct=row["window_bootstrap_95ci_e2e_pct"][0],
        window_bootstrap_95ci_e2e_high_pct=row["window_bootstrap_95ci_e2e_pct"][1],
        window_bootstrap_95ci_tail_low_pct=row["window_bootstrap_95ci_tail_pct"][0],
        window_bootstrap_95ci_tail_high_pct=row["window_bootstrap_95ci_tail_pct"][1])
        for row in relative])
    summary = dict(audited=True, cal_cells=8, cal_runs=8*len(design["cal_methods"])*design["cal_repeats"],
        calibration_scores_ms=cal_score, test_cells=24, runs=len(runs),
        rejected_cal_attempts=sum(not row["passed"] for row in cal_status["attempts"]),
        rejected_test_attempts=sum(not row["passed"] for row in test_status["attempts"]),
        selected_fixed_budget=selection["fixed"],
        selected_guarded_budget=selection["guarded"],
        aggregate=aggregate, paired=relative,
        inference="Six date/source windows; shapes and capacities are correlated within window")
    write(analysis / "summary.json", summary)
    lines = ["# 固定控制校准与独立窗口测试", "",
        "全部 8 个校准场景与 24 个测试场景通过 A/A 和逐轮文件 SHA256 审计。测试为 5 方法 × 4 重复，共 480 次正式 GPU 测量。",
        f"校准另有 {summary['rejected_cal_attempts']} 次、测试另有 {summary['rejected_test_attempts']} 次技术尝试未通过门槛；失败记录保留，不纳入正式均值。",
        "校准与测试来自同一 Azure 周轨迹，但在每个来源内按日期和请求 ID 分离；同窗口的到达形态与容量不是独立样本。", "",
        f"校准选定的原生固定预算：**{selection['fixed']}**；带相同保护的固定候选：**{selection['guarded']}**。两个预算分别在校准集上选择，测试结果不参与选择。", "",
        "| 方法 | mean TTFT (ms) | P99 TTFT (ms) | mean E2E (ms) | P99 max ITL (ms) | 抢占/场景 | 重算 token/场景 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in aggregate:
        lines.append(f"| {row['arm']} | {row['ttft_ms_mean']:.1f} | {row['ttft_ms_p99']:.1f} | {row['e2e_ms_mean']:.1f} | {row['max_itl_ms_p99']:.1f} | {row['preemptions']:.1f} | {row['recomputed_tokens']:.1f} |")
    lines += ["", "正值表示完整 SPARE 的 mean TTFT 更低；先在场景内平均四重复，再按六个源窗口成组重采样。", "",
        "| 对照 | SPARE TTFT 改善 | TTFT 六窗口 95% 区间 | E2E 改善 | P99 max ITL 改善 | TTFT 场景胜/负 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in relative:
        low, high = row["window_bootstrap_95ci_pct"]
        lines.append(f"| {row['reference']} | {row['mean_improvement_pct']:+.2f}% | [{low:+.2f}%, {high:+.2f}%] | {row['mean_e2e_improvement_pct']:+.2f}% | {row['mean_tail_improvement_pct']:+.2f}% | {row['wins']}/{row['losses']} |")
    lines += ["", "## 结论边界", ""]
    for reference, label in (("fixed_selected", "原生固定预算"),
                             ("guarded_selected", "带保护的固定候选")):
        row = next(r for r in relative if r["reference"] == reference)
        low, high = row["window_bootstrap_95ci_pct"]
        if low > 0:
            claim = "在本测试的六个源窗口上显示正向平均优势"
        elif high < 0:
            claim = "在本测试的六个源窗口上显示平均退化"
        else:
            claim = "区间跨零，未确认平均优势或退化"
        lines.append(f"- 相对{label}，SPARE 的配对改善为 {row['mean_improvement_pct']:+.2f}%（[{low:+.2f}%, {high:+.2f}%]）；{claim}。")
    lines += ["- 固定候选带有条件回退，不等同于原生固定预算。六源窗口区间属于探索性不确定性描述，不是跨数据集泛化保证。", ""]
    (analysis / "RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
