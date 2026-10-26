"""Freeze the Sarathi supplement against the original validation cohort."""
from __future__ import annotations

import json

from research.www_simulator_gpu_validation_20261018 import prepare as base


OUT = base.ROOT / "artifacts/www_simulator_gpu_validation_sarathi_20261018_r1"


def main():
    original = json.loads((base.OUT / "design.json").read_text())
    base.OUT = OUT
    base.METHODS = {2048: ("sarathi",), 6144: ("sarathi",)}
    base.main()
    path = OUT / "design.json"
    frozen = json.loads(path.read_text())
    assert frozen["samples"] == original["samples"]
    assert (frozen["horizon_ms"], frozen["warm_ms"], frozen["backlog_interval_ms"]) == (
        original["horizon_ms"], original["warm_ms"], original["backlog_interval_ms"])
    frozen["purpose"] = "Sarathi supplement to independent RTX 4090 simulator validation"
    frozen["paired_design_sha256"] = base.sha(base.ROOT / "artifacts/www_simulator_gpu_validation_20261018_r2/design.json")
    for name in ("prepare_sarathi.py", "run_sarathi.py"):
        rel = "research/www_simulator_gpu_validation_20261018/" + name
        frozen["source_hashes"][rel] = base.sha(base.ROOT / rel)
    base.write(path, frozen)


if __name__ == "__main__":
    main()
