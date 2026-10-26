"""Audit Code long-output runs and report each output band."""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_code_long_output_gpu_20261010 import common as spec
from research.www_evaluation_gpu_completion_20261003 import analyze as base

base.ARMS = spec.ARMS
base.ORACLE_ARMS = spec.ORACLE_ARMS
base.OUT = spec.OUT
base.HERE = spec.HERE


def main():
    base.main()
    design = spec.read(spec.OUT / "design.json")
    bands = {cell["key"]: cell["output_band"] for cell in design["cells"]}
    groups = defaultdict(list)
    with (spec.OUT / "analysis/runs.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups[(bands[row["cell"]], row["cell"], row["arm"])].append(float(row["ttft_ms_mean"]))
    lines = ["# Code 长输出 GPU 压力测试", "",
        "全部运行通过逐轮哈希审计后生成。513–1024 与 1025–1500 output tokens 分档；",
        "每窗口 64 个长输出请求，16 个同时到达、48 个以 5.25 req/s 到达。",
        "Code 另有 output>1500 的请求，本批只与 Conversation 共用两档比较。", "",
        "| 输出档 | 对照 | 场景 | SPARE 平均 TTFT 改善 | 胜/负 |",
        "| --- | --- | ---: | ---: | ---: |"]
    band_rows = []
    for band in ("output513_1024", "output1025_1500"):
        cells = [cell["key"] for cell in design["cells"] if cell["output_band"] == band]
        for arm in spec.ARMS[1:]:
            effects = []
            for cell in cells:
                ours = st.fmean(groups[band, cell, "arrival"])
                other = st.fmean(groups[band, cell, arm])
                effects.append(100 * (1 - ours / other))
            band_rows.append(dict(output_band=band, reference=arm,
                cells=len(effects), mean_ttft_improvement_pct=st.fmean(effects),
                wins=sum(effect > 0 for effect in effects),
                losses=sum(effect < 0 for effect in effects)))
            lines.append(f"| {band} | {arm} | {len(effects)} | "
                f"{st.fmean(effects):+.2f}% | "
                f"{sum(effect > 0 for effect in effects)}/{sum(effect < 0 for effect in effects)} |")
    base.csv_write(spec.OUT / "analysis/relative_to_arrival.csv", band_rows)
    summary = spec.read(spec.OUT / "analysis/summary.json")
    summary["relative_to_arrival"] = band_rows
    summary["relative_scope"] = "Bands kept separate; no pooled window bootstrap because dates overlap"
    spec.write(spec.OUT / "analysis/summary.json", summary)
    lines += ["", "完整逐次、逐场景结果见输出目录 `analysis/`。", ""]
    (spec.HERE / "RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
