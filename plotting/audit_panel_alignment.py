"""Portable panel geometry check used by the released plotting scripts."""

from __future__ import annotations

import json
from pathlib import Path


def require_matplotlib_panel_alignment(fig, axes=None, panel_ids=None,
                                       json_out=None, strict=True, **_options):
    fig.canvas.draw()
    panels = list(axes if axes is not None else fig.axes)
    ids = list(panel_ids) if panel_ids is not None else [str(i + 1) for i in range(len(panels))]
    if len(ids) != len(panels):
        raise ValueError("panel_ids must match axes")
    rows = []
    for panel_id, ax in zip(ids, panels):
        bounds = [float(value) for value in ax.get_position().bounds]
        valid = (0 <= bounds[0] <= 1 and 0 <= bounds[1] <= 1
                 and bounds[2] > 0 and bounds[3] > 0
                 and bounds[0] + bounds[2] <= 1.001
                 and bounds[1] + bounds[3] <= 1.001)
        rows.append({"id": str(panel_id), "bounds": bounds, "valid": valid})
    report = {"status": "pass" if all(row["valid"] for row in rows) else "fail",
              "panels": rows}
    if json_out is not None:
        target = Path(json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if strict and report["status"] != "pass":
        raise ValueError("panel extends beyond the figure bounds")
    return report
