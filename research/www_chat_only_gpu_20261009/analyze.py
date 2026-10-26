"""Audit all Conversation-only runs and report the paired effects."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_chat_only_gpu_20261009 import common as spec
from research.www_evaluation_gpu_completion_20261003 import analyze as base

base.ARMS = spec.ARMS
base.ORACLE_ARMS = spec.ORACLE_ARMS
base.OUT = spec.OUT
base.HERE = spec.HERE


def main():
    base.main()
    summary = spec.read(spec.OUT / "analysis/summary.json")
    lines = ["# Conversation-only 突发后持续到达 GPU 结果", "",
        "六个新选 Conversation 窗口 × 三档容量 × 四方法 × 四重复，完整审计后生成。",
        "该批在见到混合 Code/Conversation 结果后设计，仅能作为 Conversation 专项复验；不替换原混合批次。", "",
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
    lines += ["", "逐场景和逐次原始结果见输出目录。", ""]
    (spec.HERE / "RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
