"""Audit every completed GPU run and summarize the frozen mixed-arrival batch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_burst_background_gpu_20261009 import common as spec
from research.www_evaluation_gpu_completion_20261003 import analyze as base

base.ARMS = spec.ARMS
base.ORACLE_ARMS = spec.ORACLE_ARMS
base.OUT = spec.OUT
base.HERE = spec.HERE


def main():
    base.main()
    summary = spec.read(spec.OUT / "analysis/summary.json")
    lines = ["# 突发后持续到达 GPU 结果", "",
        "本批按预先冻结的 18 场景 × 4 方法 × 4 重复完成，全部逐轮哈希与 A/A 检查通过。",
        "前 32 个请求同时到达，余下 96 个按 5.25 req/s 到达。六个源窗口与 G2 相同，不是独立数据集。", "",
        "| 方法 | mean TTFT (ms) | mean E2E (ms) | 输出 token/s | SLO 达标率 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary["aggregate"]:
        lines.append(f"| {row['arm']} | {row['ttft_ms_mean']:.1f} | "
            f"{row['e2e_ms_mean']:.1f} | {row['output_throughput_tps']:.2f} | "
            f"{row['slo_ttft5000_maxitl200_attainment']:.1%} |")
    lines += ["", "SPARE 相对各方法的 mean TTFT，正值表示 SPARE 更快：", "",
        "| 对照 | 平均改善 | 六窗口 95% 区间 | 场景胜/平/负 |",
        "| --- | ---: | ---: | ---: |"]
    for row in summary["relative_to_arrival"]:
        lines.append(f"| {row['reference']} | "
            f"{row['arrival_mean_ttft_improvement_pct']:+.2f}% | "
            f"[{row['bootstrap_window_95ci_low_pct']:+.2f}%, "
            f"{row['bootstrap_window_95ci_high_pct']:+.2f}%] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |")
    lines += ["", "逐次原始结果、配置和 SHA256 审计见输出目录。", ""]
    (spec.HERE / "RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
