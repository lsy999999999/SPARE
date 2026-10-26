"""Freeze replays of old-rate cells under the current GPU runner version."""
from __future__ import annotations

import json

from research.www_policy_ranking_20261023.prepare import (
    ARMS, OLD, OUT, RATES, WINDOWS, frozen_write, label,
)


def main() -> None:
    count = 0
    for source, rates in RATES.items():
        for window in WINDOWS:
            sample = f"{source}_{window}"
            tasks = []
            for arm in ARMS:
                for rate in rates:
                    old = OLD / f"{label(sample, arm, rate)}.json"
                    if old.exists():
                        row = json.loads(old.read_text())
                        if row.get("status") != "complete" or not row.get("valid"):
                            raise RuntimeError(f"invalid existing run: {old}")
                        tasks.append(dict(sample=sample, arm=arm, rate_qps=rate))
            count += len(tasks)
            frozen_write(OUT / "tasks_recheck" / f"{sample}.json", dict(
                model="qwen3_8b", kv_blocks=2048, source=source,
                sample=sample, tasks=tasks))
    print(f"Frozen {count} current-runner rechecks")


if __name__ == "__main__":
    main()
