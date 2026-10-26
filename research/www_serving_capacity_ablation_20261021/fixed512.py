"""Refine fixed-512 capacity on the shared 24 Conversation/2048 windows."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from research.www_biobjective_ablation_mean_20261019 import run as old
from research.www_component_ablation_mean_20261019 import confirm

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT / "artifacts/www_component_ablation_mean_20261020_r4"
ARCHIVE = old.OUT
OUT = ROOT / "artifacts/www_serving_capacity_ablation_20261021/fixed512"
SOURCE = "conversation"
KV = 2048
ARM = "fixed512"
THETAS = (1000, 2000, 4000, 8000)


def read(path):
    return json.loads(path.read_text())


def samples():
    current = read(PRIMARY / "design.json")
    archived = read(ARCHIVE / "design.json")
    assert current["timing_sha256"] == archived["timing_sha256"]
    by_name = {sample["name"]: sample for sample in archived["samples"]}
    selected = [sample for sample in current["samples"]
                if sample["source"] == SOURCE]
    assert len(selected) == 24 and all(by_name[s["name"]] == s for s in selected)
    assert archived["arms"][ARM] == ["arrival", {"mode": "fixed", "fixed_budget": 512}]
    return selected


def rate_path(rate):
    return OUT / "rates" / old.rate_name(SOURCE, KV, ARM, rate)


def simulate(task):
    sample, rate = task
    return old.prior.simulate_one(sample, KV, ARM, rate)


def main(workers):
    selected = samples()
    wanted = {sample["name"] for sample in selected}
    original = read(ARCHIVE / "groups" / f"{SOURCE}_c{KV}_{ARM}.json")
    archived_names = {row["name"] for row in read(
        ARCHIVE / "rates" / old.rate_name(SOURCE, KV, ARM, 3))["rows"]}
    assert wanted <= archived_names
    OUT.mkdir(parents=True, exist_ok=True)
    design = {
        "source": SOURCE, "kv_blocks": KV, "arm": ARM,
        "sample_names": sorted(wanted), "theta_ms": THETAS,
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "capacity_rule": "24 paired windows; mean TTFT, mean window P99 TBT, backlog",
    }
    design_path = OUT / "design.json"
    if design_path.exists():
        assert read(design_path) == json.loads(json.dumps(design))
    else:
        design_path.write_text(json.dumps(design, indent=2) + "\n")
    scans = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=old.initialize) as pool:
        def get_rate(rate):
            target = rate_path(rate)
            if target.exists():
                data = read(target)
            else:
                archived = ARCHIVE / "rates" / target.name
                if archived.exists():
                    rows = [row for row in read(archived)["rows"]
                            if row["name"] in wanted]
                else:
                    rows = list(pool.map(simulate,
                                         ((sample, rate) for sample in selected)))
                rows.sort(key=lambda row: row["name"])
                data = {"rows": rows, "reused_from": str(archived.relative_to(ROOT))
                        if archived.exists() else None,
                        "summary": confirm.summary(rows, rate)}
                old.prior.base.write(target, data)
            assert {row["name"] for row in data["rows"]} == wanted
            assert data["summary"] == confirm.summary(data["rows"], rate)
            scans[rate] = data["summary"]

        for rate in original["rates"]:
            get_rate(rate)
        for theta in THETAS:
            low, high, _ = old.bounds(scans, theta)
            assert low > 0 and high is not None, (theta, low, high)
            while high - low > .0200001:
                rate = round((low + high) * 50) / 100
                assert low < rate < high
                get_rate(rate)
                low, high, _ = old.bounds(scans, theta)
    capacities = {}
    for theta in THETAS:
        low, high, cause = old.bounds(scans, theta)
        assert low < high and high - low <= .0200001
        capacities[str(theta)] = {
            "last_pass_qps": low, "first_fail_qps": high,
            "failure_causes": cause,
        }
    newly_simulated_rates = sum(read(rate_path(rate))["reused_from"] is None
                                for rate in scans)
    result = {"source": SOURCE, "kv_blocks": KV, "arm": ARM,
              "rates": sorted(scans), "capacities": capacities,
              "newly_simulated_rates": newly_simulated_rates,
              "newly_simulated_windows": 24 * newly_simulated_rates}
    old.prior.base.write(OUT / "group.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args().workers)
