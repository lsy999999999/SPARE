"""Prelaunch semantic and provenance checks; no GPU timing claims."""
from __future__ import annotations

import math
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_gpu_baselines_20260929.common import *
from research.www_gpu_baselines_20260929.run import peak_kv_fraction
from writing.www.code.run_simulator import simulate


def main():
    design = read(OUT / "design.json")
    assert tuple(design["methods"]) == ARMS
    assert len(design["cells"]) == 16
    assert design["expected_formal_runs"] == 832
    assert math.isclose(peak_kv_fraction(
        [dict(total_blocks=10, free_after=3)]), 2 / 3)
    assert math.isclose(peak_kv_fraction(
        [dict(total_blocks=10, usable_blocks=8, free_after=2)]), .75)
    requests = [
        dict(id="v0", prompt=128, output=4, arrival_ms=0.),
        dict(id="v1", prompt=256, output=6, arrival_ms=0.),
        dict(id="v2", prompt=384, output=8, arrival_ms=2.),
        dict(id="v3", prompt=512, output=10, arrival_ms=4.),
    ]
    history = [4, 6, 8, 10, 12, 16, 24, 32]
    results = {}
    for arm in ARMS:
        result = simulate(requests, arm=arm, capacity=128, budget=256,
                          chunk=128, profile="resource", history=history,
                          max_wall_s=60)
        assert result["complete"] and result["metrics"]["completed"] == 4
        assert result["uses_realized_output_lengths"] == (arm in ORACLE_ARMS)
        results[arm] = dict(rounds=result["metrics"]["rounds"],
                           trajectory_sha256=result["trajectory_sha256"])
    write(OUT / "validation.json", dict(passed=True, created=time.time(),
        arms=list(ARMS), smoke=results, source_hashes=source_hashes(),
        scope="Native V1 CPU allocation and policy semantics only; no CUDA timing"))
    print(json.dumps(read(OUT / "validation.json"), indent=2))


if __name__ == "__main__":
    main()
