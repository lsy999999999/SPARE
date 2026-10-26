"""Validate long-output inputs and actual scheduler paths on the CPU allocator."""
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_long_output_gpu_20261010.common import *
from research.www_evaluation_gpu_completion_20261003.scheduler import GPUBaselineScheduler
from research.www_evaluation_completion_20261003 import simulate as sim


def main():
    design = read(OUT / "design.json")
    assert design["source_hashes"] == source_hashes()
    assert len(design["cells"]) == 27 and design["methods"] == list(ARMS)
    for cell in design["cells"]:
        rows = read(OUT / cell["input"])
        assert len(rows) == 64
        assert digest(OUT / cell["input"]) == cell["input_sha256"]
        assert digest(ROOT / cell["source_input"]) == cell["source_input_sha256"]
        assert digest(OUT / cell["history_file"]) == cell["history_file_sha256"]
        minimum, maximum = ((513, 1024) if cell["output_band"] == "output513_1024"
                            else (1025, 1500))
        assert all(row["prompt"] > 0 and minimum <= row["output"] <= maximum
                   and row["prompt"] + row["output"] <= 8000 for row in rows)
        assert all(row["arrival_ms"] == 0 for row in rows[:16])
        assert all(abs(row["arrival_ms"] - (index - 15) * 1000 / 5.25) < 1e-8
                   for index, row in enumerate(rows) if index >= 16)

    rows = [dict(id=f"long{i}", prompt=256 + i * 160,
                 output=620 if i % 2 == 0 else 1150,
                 arrival_ms=0. if i < 2 else (i - 1) * 200.)
            for i in range(6)]
    checks = {}
    with patch.object(sim, "FormulaScheduler", GPUBaselineScheduler), \
         patch.object(sim, "ComparisonScheduler", GPUBaselineScheduler), \
         patch.object(sim, "BudgetScheduler", GPUBaselineScheduler):
        for arm in (*ARMS, "native"):
            result = sim.simulate(rows,
                arm=arm if arm in ("srp", "native") else "arrival",
                capacity=192, history=[4, 8, 16, 24, 32, 64],
                profile="resource", policy_options=OPTIONS.get(arm, {}))
            metrics = result["metrics"]
            assert result["complete"] and metrics["completed"] == len(rows)
            expected = sum(row["prompt"] + row["output"] - 1 for row in rows)
            assert metrics["executed_tokens"] - metrics["recomputed_tokens"] == expected
            checks[arm] = dict(rounds=metrics["rounds"],
                               trajectory=result["trajectory_sha256"])
    write(OUT / "validation.json", dict(passed=True, created=time.time(),
        source_hashes=source_hashes(), checks=checks,
        scope="64-request long-output inputs; native allocator completion and conservation"))
    print(dict(passed=True, methods=len(checks), cells=len(design["cells"])))


if __name__ == "__main__":
    main()
