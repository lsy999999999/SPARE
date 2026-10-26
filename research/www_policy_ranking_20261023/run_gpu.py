"""Run the existing Qwen3-8B GPU replay into the ranking-specific artifact."""
from pathlib import Path

from research.www_cross_model_20261021 import run_gpu as base


base.OUT = base.ROOT / "artifacts/www_policy_ranking_fidelity_20261023_r1"

if __name__ == "__main__":
    base.main()
