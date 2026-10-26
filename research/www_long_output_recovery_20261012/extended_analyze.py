"""Audit a shadow matrix using the extended-timeout cell without changing the frozen one."""
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_long_output_gpu_20261010 import common as spec
from research.www_evaluation_gpu_completion_20261003 import analyze as base

HERE = Path(__file__).resolve().parent
ORIGINAL = spec.OUT
VIEW = ORIGINAL / "supplemental_extended_20261012"
CELL = "conversation_output1025_1500_20240512_burst_background_c768"


def main():
    design = spec.read(ORIGINAL / "design.json")
    for name in ("design.json", "manifest.json", "attempts"):
        link = VIEW / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to((ORIGINAL / name).resolve(), target_is_directory=name == "attempts")
    for cell in design["cells"]:
        name = cell["key"]
        source = ORIGINAL / "groups" / f"{name}.json"
        record = spec.read(source)
        if name == CELL:
            target = ORIGINAL / "attempts" / CELL / "a5" / "attempt_status.json"
            record = spec.read(target)
        if record["state"] != "complete":
            raise RuntimeError(f"incomplete supplemental group: {name}")
        spec.write(VIEW / "groups" / f"{name}.json", record)
    base.OUT = VIEW
    base.HERE = HERE
    base.ARMS = spec.ARMS
    base.ORACLE_ARMS = spec.ORACLE_ARMS
    base.main()
    bands = {cell["key"]: cell["output_band"] for cell in design["cells"]}
    groups = {}
    with (VIEW / "analysis/runs.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault((bands[row["cell"]], row["cell"], row["arm"]), []).append(
                float(row["ttft_ms_mean"]))
    lines = ["# Conversation 长输出：宽时限补充 GPU 结果", "",
        "原协议在低 KV 容量场景中多次触发 SRP 单次运行时限，完整矩阵未通过原验收。",
        "本补充只对该失败场景放宽单次运行时限至 21600 秒；输入、方法、顺序、稳定性门槛和计时逻辑沿用原实验。",
        "其他已完成场景仍使用原 2400 秒时限。完整结果通过逐轮哈希审计；这是协议变更后的补充矩阵，",
        "不能改写为原预设矩阵通过。两档来自同一周轨迹，重合日期不视为独立数据集。", "",
        "| 输出档 | 对照 | 场景 | SPARE 平均 TTFT 改善 | 胜/负 |",
        "| --- | --- | ---: | ---: | ---: |"]
    for band in ("output513_1024", "output1025_1500"):
        cells = [cell["key"] for cell in design["cells"] if cell["output_band"] == band]
        for arm in spec.ARMS[1:]:
            effects = []
            for cell in cells:
                ours = st.fmean(groups[band, cell, "arrival"])
                other = st.fmean(groups[band, cell, arm])
                effects.append(100 * (1 - ours / other))
            lines.append(f"| {band} | {arm} | {len(effects)} | {st.fmean(effects):+.2f}% | "
                         f"{sum(x > 0 for x in effects)}/{sum(x < 0 for x in effects)} |")
    lines += ["", "逐次和逐场景审计结果见 `artifacts/www_long_output_gpu_20261010_r1/"
              "supplemental_extended_20261012/analysis/`；原超时运行及受干扰尝试仍完整保留。", ""]
    (HERE / "SUPPLEMENTAL_RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
