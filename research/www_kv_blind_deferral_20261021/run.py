"""Matched 24-window capacity scan for the KV-blind deferral control."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from research.www_component_ablation_mean_20261019 import confirm, run as pilot
from research.www_kv_blind_deferral_20261021 import policy

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT / "artifacts/www_component_ablation_mean_20261020_r4"
OUT = ROOT / "artifacts/www_kv_blind_deferral_20261021_r3"
ARM = "kv_blind"
READ = pilot.old.prior.base.read
WRITE = pilot.old.prior.base.write
DIGEST = pilot.old.prior.base.digest


def prepare():
    reference = READ(PRIMARY / "design.json")
    samples = reference["samples"]
    assert len(samples) == 48
    assert all(sum(s["source"] == source for s in samples) == 24
               for source in pilot.SOURCES)
    design = dict(samples=samples, sources=pilot.SOURCES,
                  kv_blocks=pilot.KVS, theta_ms=pilot.THETA_MS,
                  windows_per_source=24, arm=ARM,
                  decision_inputs=("request progress", "queue order",
                                   "arrival token rate", "token budget",
                                   "chunk", "concurrency", "model length"),
                  excluded_decision_inputs=("free KV blocks", "total KV blocks",
                                            "request-held KV blocks", "KV growth",
                                            "physical block savings"),
                  token_saving_threshold="max(1, chunk - ceil(arrival_tokens_per_second))",
                  capacity_rule=reference["capacity_rule"],
                  reference_design_sha256=DIGEST(PRIMARY / "design.json"),
                  policy_sha256=DIGEST(Path(policy.__file__)),
                  gpu_execution=False)
    path = OUT / "design.json"
    if path.exists():
        assert READ(path) == json.loads(json.dumps(design))
    else:
        WRITE(path, design)
    return samples


def initialize():
    pilot.initialize()
    pilot.old.prior.ARMS[ARM] = ("arrival", {"mode": ARM})
    from research.www_evaluation_completion_20261003 import scheduler
    scheduler.select = policy.select
    confirm.OUT = OUT
    confirm.get_rate = get_rate


def get_rate(source, kv, arm, rate, samples):
    assert arm == ARM
    target = OUT / "rates" / pilot.old.rate_name(source, kv, arm, rate)
    wanted = {sample["name"] for sample in samples}
    if target.exists():
        data = READ(target)
    else:
        checkpoint = OUT / "checkpoints" / target.name
        rows = READ(checkpoint)["rows"] if checkpoint.exists() else []
        found = {row["name"] for row in rows}
        assert len(rows) == len(found) and found <= wanted
        for sample in samples:
            if sample["name"] not in found:
                rows.append(pilot.old.prior.simulate_one(sample, kv, arm, rate))
                WRITE(checkpoint, dict(rows=rows))
        rows.sort(key=lambda row: row["name"])
        data = dict(rows=rows, summary=confirm.summary(rows, rate),
                    newly_simulated=len(rows))
        WRITE(target, data)
    assert {row["name"] for row in data["rows"]} == wanted
    assert data["summary"] == confirm.summary(data["rows"], rate)
    return data["summary"]


def smoke():
    from research.v1_budget_20260922.policy import State
    rows = [State(str(i), 0, 2048, 2048, 0, blocks, True)
            for i, blocks in enumerate((0, 2, 10, 20))]
    altered = [State(r.rid, r.computed, r.tokens, r.prompt, r.outputs,
                     r.blocks + 100, r.running) for r in rows]
    first = policy.decide(policy.request_views(rows), 1024, 256, 32, 4096, 64)
    second = policy.decide(policy.request_views(altered), 1024, 256, 32, 4096, 64)
    assert not hasattr(policy.request_views(rows)[0], "blocks")
    assert (first["budget"], first["chunk"]) == (256, 256)
    assert (first["budget"], first["chunk"]) == (second["budget"], second["chunk"])
    for state in (rows, altered):
        for free, total, block_size in ((0, 1024, 16), (128, 2048, 16),
                                        (4096, 8192, 32)):
            choice = policy.select(state, 1024, 256, free, total, block_size,
                                   32, 4096, recent={"arrival_rate": 64},
                                   options={"mode": ARM})
            assert (choice["budget"], choice["chunk"]) == (256, 256)
    samples = prepare()
    initialize()
    sample = next(s for s in samples if s["source"] == "conversation")
    result = pilot.old.prior.simulate_one(sample, 2048, ARM, 2.)
    assert result["valid"] and result["interventions"] > 0
    WRITE(OUT / "smoke.json", dict(passed=True,
                                   kv_allocation_invariant=True,
                                   sample=result["name"],
                                   interventions=result["interventions"]))
    print("KV-blind decision invariance and one-window replay passed")


def main(workers):
    samples = prepare()
    assert READ(OUT / "smoke.json")["passed"]
    tasks = [(source, kv, ARM,
              [sample for sample in samples if sample["source"] == source])
             for source in pilot.SOURCES for kv in pilot.KVS]
    pending = [task for task in tasks if not (
        OUT / "groups" / f"{task[0]}_c{task[1]}_{ARM}.json").exists()]
    start = time.monotonic()
    errors = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        futures = {pool.submit(confirm.run_group, task): task[:3]
                   for task in pending}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append(dict(group=futures[future], error=repr(exc)))
            status = dict(completed=len(tasks)-len(pending)
                          + sum(f.done() and f.exception() is None for f in futures),
                          total=len(tasks), errors=errors,
                          elapsed_s=round(time.monotonic()-start, 1))
            WRITE(OUT / "status.json", status)
            print(status, flush=True)
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    smoke() if args.smoke else main(args.workers)
