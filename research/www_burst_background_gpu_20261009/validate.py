"""Check frozen inputs and four methods on the native CPU allocator."""
import time
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_burst_background_gpu_20261009.common import *
from research.www_evaluation_gpu_completion_20261003.scheduler import GPUBaselineScheduler
from research.www_evaluation_completion_20261003 import simulate as sim
from writing.www.code.run_simulator import simulate as frozen_simulate


def main():
    design = read(OUT / "design.json")
    assert design["source_hashes"] == source_hashes()
    assert len(design["cells"]) == 18 and design["methods"] == list(ARMS)
    for cell in design["cells"]:
        rows = read(OUT / cell["input"])
        assert digest(OUT / cell["input"]) == cell["input_sha256"]
        assert len(rows) == 128 and all(r["arrival_ms"] == 0 for r in rows[:32])
        assert all(abs(r["arrival_ms"] - (i - 31) * 1000 / 5.25) < 1e-8
                   for i, r in enumerate(rows) if i >= 32)
        assert digest(ROOT / cell["source_input"]) == cell["source_input_sha256"]
        assert digest(OUT / cell["history_file"]) == cell["history_file_sha256"]
    rows = [dict(id=f"r{i}", prompt=(128, 800, 1800, 2500)[i % 4] + i,
                 output=24, arrival_ms=0. if i < 5 else (i - 4) * 40.)
            for i in range(20)]
    history = [4, 8, 16, 24, 32, 64]
    checks = {}
    with patch.object(sim, "FormulaScheduler", GPUBaselineScheduler), \
         patch.object(sim, "ComparisonScheduler", GPUBaselineScheduler), \
         patch.object(sim, "BudgetScheduler", GPUBaselineScheduler):
        for arm in (*ARMS, "native"):
            options = OPTIONS.get(arm, {})
            result = sim.simulate(rows, arm=arm if arm in ("srp", "native") else "arrival",
                capacity=192, history=history, profile="resource",
                policy_options=options)
            metrics = result["metrics"]
            assert result["complete"] and metrics["completed"] == len(rows)
            expected = sum(row["prompt"] + row["output"] - 1 for row in rows)
            assert metrics["executed_tokens"] - metrics["recomputed_tokens"] == expected
            if arm == "arrival":
                frozen = frozen_simulate(rows, capacity=192, profile="resource",
                                         history=history)
                assert result["trajectory_sha256"] == frozen["trajectory_sha256"]
            checks[arm] = dict(rounds=metrics["rounds"],
                               trajectory=result["trajectory_sha256"])
    write(OUT / "validation.json", dict(passed=True, created=time.time(),
        source_hashes=source_hashes(), checks=checks,
        scope="input hashes and shape; native CPU allocator completion and conservation"))
    print(dict(passed=True, methods=len(checks)))


if __name__ == "__main__":
    main()
