"""Rebuild the comparison figures from the audited, frozen summary.json.

Usage: python3 research/v1_paper_comparison_20260922_charts.py
Requires Pillow and the system Noto Sans CJK font.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "artifacts/v1_paper_comparison_multi_20260922_r3/summary.json"
OUT = Path(__file__).resolve().parent / "v1_paper_comparison_20260922_charts"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

INK = "#172B31"
MUTED = "#52636A"
GRID = "#DDE5E6"
PAPER = "#FFFFFF"
COLORS = {
    "budget": "#087C64",
    "srp": "#61717A",
    "sarathi_v1": "#BA6530",
    "past_future_v1": "#3565A7",
}
NAMES = {
    "budget": "当前 A",
    "srp": "调优 SRP",
    "sarathi_v1": "Sarathi 规则版",
    "past_future_v1": "Past-Future 规则版",
}
METHODS = tuple(COLORS)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size, index=2)


def label(draw: ImageDraw.ImageDraw, x: int, y: int, value: str,
          size: int = 27, color: str = INK, bold: bool = False) -> None:
    draw.text((x, y), value, font=font(size, bold), fill=color)


def base(width: int, height: int, title: str, subtitle: str):
    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    label(draw, 72, 38, title, 42, bold=True)
    label(draw, 72, 103, subtitle, 24, color=MUTED)
    draw.line((72, 157, width - 72, 157), fill=GRID, width=2)
    return image, draw


def axes(draw: ImageDraw.ImageDraw, left: int, right: int,
         top: int, bottom: int, maximum: int, step: int) -> None:
    for tick in range(0, maximum + 1, step):
        x = round(left + (right - left) * tick / maximum)
        draw.line((x, top, x, bottom), fill=GRID, width=2 if tick == 0 else 1)
        caption = str(tick)
        width = draw.textbbox((0, 0), caption, font=font(21))[2]
        label(draw, x - width // 2, top - 36, caption, 21, color=MUTED)


def ttft_by_regime(summary: dict) -> Image.Image:
    image, draw = base(1200, 1190, "平均首 token 延迟", "三种负载分别使用独立横轴刻度；越短越好")
    metrics = {(m["regime"], m["arm"]): m for m in summary["metrics"]}
    regimes = (
        ("queue", "普通队列", 2, 1),
        ("burst", "突发 · 完整 KV", 8, 2),
        ("restricted", "突发 · 受限 KV", 70, 10),
    )
    left, right = 440, 996
    for index, (key, name, maximum, step) in enumerate(regimes):
        top = 190 + index * 319
        label(draw, 72, top, name, 32, bold=True)
        label(draw, 948, top + 5, "TTFT / 秒", 20, color=MUTED)
        axes(draw, left, right, top + 78, top + 282, maximum, step)
        for row, method in enumerate(METHODS):
            value = metrics[key, method]["mean_ttft_ms"] / 1000
            y = top + 103 + row * 47
            label(draw, 86, y - 8, NAMES[method], 26,
                  bold=method == "budget")
            draw.rounded_rectangle(
                (left, y, left + (right - left) * value / maximum, y + 28),
                radius=4, fill=COLORS[method],
            )
            label(draw, 1014, y - 6, f"{value:.3f}", 25, bold=method == "budget")
    label(draw, 72, 1143, "四个请求块 × 两次重复 / 方法 / 场景", 22, color=MUTED)
    return image


def per_block(summary: dict) -> Image.Image:
    image, draw = base(1200, 1260, "受限 KV：同块逐项比较", "同一请求块的四种方法固定在同一张 GPU 上；越短越好")
    pairs = summary["comparisons"]["restricted"]
    by_method = {
        name: {p["block"]: p for p in data["pairs"]}
        for name, data in pairs.items()
    }
    groups = (
        ("compare_code_0", "代码 0", 2),
        ("compare_code_1", "代码 1", 2),
        ("compare_conversation_0", "对话 0", 6),
        ("compare_conversation_1", "对话 1", 7),
    )
    left, right, maximum = 431, 1050, 100
    for index, (block, name, gpu) in enumerate(groups):
        top = 194 + index * 249
        label(draw, 72, top, f"{name}   GPU {gpu}", 31, bold=True)
        label(draw, 1048, top + 4, "秒", 20, color=MUTED)
        axes(draw, left, right, top + 78, top + 225, maximum, 20)
        for row, method in enumerate(METHODS):
            reference = by_method["srp"][block]
            value = (reference["a_ttft_ms"] if method == "budget"
                     else by_method[method][block]["baseline_ttft_ms"]) / 1000
            y = top + 88 + row * 36
            label(draw, 84, y - 8, NAMES[method], 24,
                  bold=method == "budget")
            bar_end = left + (right - left) * value / maximum
            draw.rounded_rectangle((left, y, bar_end, y + 20),
                                   radius=3, fill=COLORS[method])
            label(draw, round(bar_end) + 12, y - 9, f"{value:.2f}", 23,
                  bold=method == "budget")
        if index < len(groups) - 1:
            draw.line((72, top + 240, 1128, top + 240), fill=GRID, width=2)
    label(draw, 72, 1215, "每根横条为该块两次重复的平均 TTFT", 22, color=MUTED)
    return image


def throughput(summary: dict) -> Image.Image:
    image, draw = base(1200, 666, "受限 KV：输出吞吐", "每次运行的输出 token/s 取算术平均；越长越好")
    metrics = {m["arm"]: m for m in summary["metrics"]
               if m["regime"] == "restricted"}
    left, right, maximum = 440, 1005, 140
    axes(draw, left, right, 244, 562, maximum, 20)
    for row, method in enumerate(METHODS):
        value = metrics[method]["throughput_tokens_s"]
        y = 267 + row * 73
        label(draw, 86, y - 8, NAMES[method], 27,
              bold=method == "budget")
        bar_end = left + (right - left) * value / maximum
        draw.rounded_rectangle((left, y, bar_end, y + 36),
                               radius=4, fill=COLORS[method])
        label(draw, round(bar_end) + 13, y - 4, f"{value:.2f}", 27,
              bold=method == "budget")
    label(draw, 72, 611, "四个请求块 × 两次重复 / 方法", 22, color=MUTED)
    return image


def main() -> None:
    summary = json.loads(RESULT.read_text())
    assert summary["complete"] and summary["all_requests_completed"]
    assert not summary["failed_runs"] and summary["expected"]["compare"] == 96
    assert {(m["regime"], m["arm"]) for m in summary["metrics"]} == {
        (regime, arm) for regime in ("queue", "burst", "restricted")
        for arm in METHODS
    }
    OUT.mkdir(exist_ok=True)
    for filename, image in (
        ("ttft_by_regime.png", ttft_by_regime(summary)),
        ("ttft_restricted_by_block.png", per_block(summary)),
        ("throughput_restricted.png", throughput(summary)),
    ):
        image.save(OUT / filename, optimize=True)
        print(OUT / filename)


if __name__ == "__main__":
    main()
