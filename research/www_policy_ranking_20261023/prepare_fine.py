"""Freeze a targeted 0.25-QPS Conversation comparison of SPARE and SRP."""
from __future__ import annotations

from research.www_policy_ranking_20261023.prepare import OUT, WINDOWS, frozen_write


RATES = (2.25, 2.75, 3.25)
ARMS = ("arrival", "srp")


def main() -> None:
    frozen_write(OUT / "fine_design.json", dict(
        model="qwen3_8b", kv_blocks=2048, source="conversation",
        windows=list(WINDOWS), arms=list(ARMS), rates_qps=list(RATES),
        selection="SPARE versus strongest baseline SRP, where the original simulator predicts separation",
        original_design="design.json",
    ))
    for window in WINDOWS:
        sample = f"conversation_{window}"
        frozen_write(OUT / "tasks_fine" / f"{sample}.json", dict(
            model="qwen3_8b", kv_blocks=2048, source="conversation",
            sample=sample,
            tasks=[dict(sample=sample, arm=arm, rate_qps=rate)
                   for rate in RATES for arm in ARMS],
        ))
    print(f"Frozen {len(WINDOWS) * len(RATES) * len(ARMS)} fine-grid cells")


if __name__ == "__main__":
    main()
