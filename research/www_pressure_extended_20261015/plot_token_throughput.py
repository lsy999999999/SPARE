"""Plot emitted output-token throughput from matched 0-120 second replays."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from research.www_pressure_extended_20261015.plot_intro_comparison import (
    DATA, METHODS, RATES, plot,
)

SOURCE = DATA / "token_throughput_120s"


def assemble():
    design = json.loads((SOURCE / "design.json").read_text())
    files = list((SOURCE / "runs").glob("*.json"))
    if len(files) != design["tasks"]:
        raise RuntimeError(f"incomplete token replay: {len(files)}/{design['tasks']}")
    data = defaultdict(list)
    for path in files:
        row = json.loads(path.read_text())
        if not row["valid"] or len(row["windows"]) != 2:
            raise RuntimeError(f"invalid replay: {path}")
        mid, end = row["windows"]
        if (mid["at_ms"], end["at_ms"]) != (60000., 120000.):
            raise RuntimeError(f"wrong replay checkpoints: {path}")
        for checkpoint in row["windows"]:
            if not (checkpoint["completed_output_tokens"] <=
                    checkpoint["emitted_output_tokens"]):
                raise RuntimeError(f"invalid token accounting: {path}")
        key = row["source"], row["capacity"], row["arm"], row["rate"]
        data[key].append(row)

    summary = []
    for source in ("code", "conversation"):
        for capacity in (768, 4096):
            reference_samples = None
            reference_arrivals = {}
            for arm, label, *_ in METHODS:
                for rate in RATES:
                    rows = data[source, capacity, arm, rate]
                    names = {r["sample"] for r in rows}
                    if len(rows) != 3 or len(names) != 3:
                        raise RuntimeError(f"missing matched token runs: {source} {capacity} {arm} {rate}")
                    if reference_samples is None:
                        reference_samples = names
                    if names != reference_samples:
                        raise RuntimeError("token methods used different test windows")
                    for row in rows:
                        arrivals = tuple(w["arrivals"] for w in row["windows"])
                        sample_key = row["sample"], rate
                        if (sample_key in reference_arrivals and
                                reference_arrivals[sample_key] != arrivals):
                            raise RuntimeError("token methods used different arrivals")
                        reference_arrivals[sample_key] = arrivals
                    tokens = np.array([r["windows"][1]["emitted_output_tokens"] / 120.
                                       for r in rows])
                    completed_only = np.array([
                        r["windows"][1]["completed_output_tokens"] / 120. for r in rows])
                    summary.append(dict(source=source, kv_blocks=capacity, method=label,
                                        arm=arm, arrival_qps=rate, windows=3,
                                        start_s=0, end_s=120,
                                        output_tokens_tps=float(tokens.mean()),
                                        output_tokens_tps_sd=float(tokens.std(ddof=1)),
                                        completed_only_tokens_tps=float(completed_only.mean()),
                                        unfinished_request_tokens_tps=float((tokens-completed_only).mean())))
    if len(summary) != len(METHODS)*4*len(RATES):
        raise RuntimeError("unexpected token summary size")
    return summary


def main():
    rows = assemble()
    with (DATA / "intro_output_tokens_0_120s.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, "output_tokens_tps", "arrival_pressure_intro_output_tokens_from0",
         "Output tokens/s (0-120 s)")


if __name__ == "__main__":
    main()
