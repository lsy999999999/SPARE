"""Audit a C matrix with a missing A/A-rejected cell for descriptive use only."""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_plan_abc_20261008 import analyze_c as full


def pct(base: float, candidate: float) -> float:
    return 100 * (base - candidate) / base


def main() -> None:
    out = full.OUT
    design = full.read(out / "design.json")
    manifest = full.read(out / "manifest.json")
    assert full.digest(out / "design.json") == manifest["design_sha256"]
    assert full.digest(full.ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1/design.json") == design["old_design_sha256"]
    for name, expected in design["source_hashes"].items():
        assert full.digest(full.ROOT / "research/www_plan_abc_20261008" / name) == expected, name
    for cell in design["cells"]:
        assert full.digest(out / cell["input"]) == cell["input_sha256"]
        assert full.digest(out / cell["history"]) == cell["history_sha256"]
    selection = full.read(out / "selection.json")
    assert selection["passed"] and full.read(out / "cal_status.json")["passed"]
    full.audit_calibration(design, selection)
    status = full.read(out / "status.json")
    test_status = full.read(out / "test_status.json")
    assert status["state"] == "finished_with_gaps"
    assert not test_status["passed"] and test_status["required"] == 24

    cells = [cell for cell in design["cells"] if cell["stage"] == "test"]
    accepted = [cell for cell in cells if (out / "groups" / (cell["key"] + ".json")).exists()]
    missing = [cell for cell in cells if cell not in accepted]
    assert len(accepted) == test_status["accepted"]
    assert len(missing) == 1 and missing[0]["key"] == "conversation_test1_steady_c2048"

    rows = []
    for cell in accepted:
        group = full.read(out / "groups" / (cell["key"] + ".json"))
        assert group["state"] == "complete" and group["stage"] == "test"
        folder = out / group["folder"]
        full.audit_aa(folder, cell, design)
        for arm in design["test_methods"]:
            runs = [full.audit_run(folder, f"{arm}_r{i}", cell, arm, i)[0]
                    for i in range(design["test_repeats"])]
            assert all(run["recorded_at"] > selection["created"] for run in runs)
            row = dict(cell=cell["key"], window=cell["window"], shape=cell["shape"],
                       capacity=cell["capacity"], arm=arm)
            row.update({metric: st.fmean(run["metrics"][metric] for run in runs)
                        for metric in full.METRICS})
            rows.append(row)
    assert len(rows) == len(accepted) * 5
    by_cell = {(row["cell"], row["arm"]): row for row in rows}
    aggregate = {}
    for arm in design["test_methods"]:
        group = [row for row in rows if row["arm"] == arm]
        aggregate[arm] = {metric: st.fmean(row[metric] for row in group)
                          for metric in ("ttft_ms_mean", "e2e_ms_mean", "max_itl_ms_p99")}
    paired = []
    for arm in design["test_methods"]:
        if arm == "spare":
            continue
        values = defaultdict(list)
        windows = defaultdict(list)
        for cell in accepted:
            base = by_cell[cell["key"], arm]
            spare = by_cell[cell["key"], "spare"]
            for name, metric in (("ttft", "ttft_ms_mean"), ("e2e", "e2e_ms_mean"),
                                 ("tail", "max_itl_ms_p99")):
                value = pct(base[metric], spare[metric])
                values[name].append(value)
                if name == "ttft":
                    windows[cell["window"]].append(value)
        paired.append(dict(reference=arm, accepted_cells=len(accepted),
            descriptive_ttft_improvement_pct=st.fmean(values["ttft"]),
            descriptive_e2e_improvement_pct=st.fmean(values["e2e"]),
            descriptive_tail_improvement_pct=st.fmean(values["tail"]),
            ttft_wins=sum(value > 0 for value in values["ttft"]),
            ttft_losses=sum(value < 0 for value in values["ttft"]),
            window_cell_counts={window: len(group) for window, group in windows.items()},
            window_ttft_improvement_pct={window: st.fmean(group)
                                         for window, group in windows.items()}))

    rejected = [row for row in test_status["attempts"] if not row["passed"]]
    aa = []
    for attempt in range(3):
        folder = out / "attempts" / missing[0]["key"] / f"a{attempt}"
        gate = full.read(folder / "aa.json")
        assert not gate["passed"]
        aa.append(dict(attempt=attempt, relative_spread=gate["relative_spread"]))
    analysis = out / "analysis"
    full.csv_write(analysis / "partial_cell_means.csv", rows)
    summary = dict(confirmatory=False, reason="One prespecified test cell failed all three A/A attempts",
        accepted_cells=len(accepted), required_cells=24, missing_cells=[cell["key"] for cell in missing],
        audited_formal_runs=len(accepted)*5*design["test_repeats"],
        selected_fixed_budget=selection["fixed"], selected_guarded_budget=selection["guarded"],
        rejected_cal_attempts=sum(not row["passed"] for row in full.read(out / "cal_status.json")["attempts"]),
        rejected_test_attempts=len(rejected), missing_cell_aa=aa,
        aggregate_absolute=aggregate, paired_descriptive=paired)
    full.write(analysis / "partial_summary.json", summary)
    lines = ["# C：未完成的固定控制测试（探索性）", "",
        f"原定 24 个测试场景中，{len(accepted)} 个通过 A/A 与逐轮文件审计；缺少 `{missing[0]['key']}`。",
        "该场景三次 SRP A/A 均未达到预设 5% TTFT/总时长门槛。原方案只允许两次技术重试，故 C 不满足完整矩阵的确认性分析条件。", "",
        f"校准仍在独立日期完成，选择原生固定预算 {selection['fixed']}、带保护固定候选 {selection['guarded']}。",
        f"其余 {len(accepted)} 场景的 {summary['audited_formal_runs']} 次正式运行均通过逐轮 SHA256、完成数、A/A 与先校准后测试的时间审计。",
        f"校准另有 {summary['rejected_cal_attempts']} 次、测试另有 {len(rejected)} 次失败技术尝试，原始文件全部保留。", "",
        "下表仅描述已接受的场景，不是预先设计的 24 场景估计；缺失场景所在窗口只有三个容量/到达组合，不能用此表断言选择性控制优于固定预算。", "",
        "| 方法 | mean TTFT (ms) | mean E2E (ms) | P99 max ITL (ms) |",
        "| --- | ---: | ---: | ---: |"]
    for arm, metrics in aggregate.items():
        lines.append(f"| {arm} | {metrics['ttft_ms_mean']:.1f} | {metrics['e2e_ms_mean']:.1f} | {metrics['max_itl_ms_p99']:.1f} |")
    lines += ["", "逐场景配对的描述性百分比（正值表示 SPARE 较低）：", "",
        "| 对照 | mean TTFT 差异 | mean E2E 差异 | TTFT 胜/负 |",
        "| --- | ---: | ---: | ---: |"]
    for row in paired:
        lines.append(f"| {row['reference']} | {row['descriptive_ttft_improvement_pct']:+.2f}% | {row['descriptive_e2e_improvement_pct']:+.2f}% | {row['ttft_wins']}/{row['ttft_losses']} |")
    lines += ["", "原生固定与带保护固定对照的六窗口 TTFT 差异：", "",
        "| 窗口 | 场景数 | 原生固定 256 | 带保护固定 256 |",
        "| --- | ---: | ---: | ---: |"]
    native = next(row for row in paired if row["reference"] == "fixed_selected")
    guarded = next(row for row in paired if row["reference"] == "guarded_selected")
    for window in sorted(native["window_cell_counts"]):
        lines.append(f"| {window} | {native['window_cell_counts'][window]} | {native['window_ttft_improvement_pct'][window]:+.2f}% | {guarded['window_ttft_improvement_pct'][window]:+.2f}% |")
    lines += ["", "P99 max ITL 的逐场景相对百分比会被小分母放大，因此上表给出场景等权绝对均值。缺失和失败尝试的详情见 `partial_summary.json` 与 `attempts/`。", ""]
    (analysis / "PARTIAL_RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
