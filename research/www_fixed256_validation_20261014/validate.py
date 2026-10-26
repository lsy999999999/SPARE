"""Native V1 CPU allocator smoke test for the frozen explanation matrix."""
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_fixed256_explanation_20261014 import common as spec
sys.path.insert(0, str(spec.OUT / "source"))
from research.www_evaluation_gpu_completion_20261003 import scheduler as baseline
from research.www_evaluation_completion_20261003 import simulate as sim

original_select = baseline.select
from research.www_fixed256_explanation_20261014.scheduler import ExplanationScheduler


def main():
    design = spec.read(spec.OUT / "design.json")
    if design["source_hashes"] != spec.source_hashes():
        raise RuntimeError("frozen source changed")
    requests = [dict(id=f"r{i:03}", prompt=(128, 800, 1800, 2500)[i % 4] + i,
                     output=12, arrival_ms=float(i // 5) * 40)
                for i in range(20)]
    results = {}
    with patch.object(sim, "FormulaScheduler", ExplanationScheduler):
        baseline.select = original_select
        reference = sim.simulate(requests, arm="arrival", capacity=192,
                                 profile="resource", history=[4, 8, 16, 24])
        try:
            from research.www_fixed256_explanation_20261014.policy import select
            baseline.select = select
            for arm in spec.ARMS:
                result = sim.simulate(requests, arm="arrival", capacity=192,
                    profile="resource", history=[4, 8, 16, 24],
                    policy_options=spec.OPTIONS[arm])
                metrics = result["metrics"]
                if not result["complete"] or metrics["completed"] != len(requests):
                    raise RuntimeError(f"incomplete CPU allocator replay: {arm}")
                expected = sum(row["prompt"] + row["output"] - 1
                               for row in requests)
                if metrics["executed_tokens"] - metrics["recomputed_tokens"] != expected:
                    raise RuntimeError(f"token conservation failed: {arm}")
                results[arm] = dict(rounds=metrics["rounds"],
                    preemptions=metrics["preemptions"],
                    trajectory_sha256=result["trajectory_sha256"])
            if results["full"]["trajectory_sha256"] != reference["trajectory_sha256"]:
                raise RuntimeError("Full differs from frozen baseline on CPU allocator")
        finally:
            baseline.select = original_select
    spec.write(spec.OUT / "validation.json", dict(passed=True,
        created=time.time(), source_hashes=spec.source_hashes(),
        checks=results, original_full_sha256=reference["trajectory_sha256"],
        scope="native V1 CPU allocator semantics; no GPU latency claim"))
    print("validated five variants and Full trajectory parity")


if __name__ == "__main__":
    main()
