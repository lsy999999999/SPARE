"""Replay a frozen KV-pressure motif with and without completion protection."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

from research.www_capacity_pareto_20261015 import run as base
from research.www_component_ablation_mean_20261019 import policy

ROOT = base.ROOT
OUT = ROOT / "artifacts/www_protection_stress_20261020_r3"
CAPACITY = 112
HEAVY_OFFSET_MS = 65.
HORIZON_MS = 300000.
WARM_MS = 100000.
MID_MS = 200000.
THETA_MS = 2000.
THETA_GRID_MS = (1000., 1500., 2000., 4000., 8000.)
RATES = (.8, 1., 1.1, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24,
         1.26, 1.28, 1.3, 1.32, 1.4)
BACKGROUND_PROMPTS = (384, 400, 416)
ARMS = {"full": {}, "no_finish_guard": {"protect_finish": False}}
MICRO_ARMS = dict(ARMS, no_decode_guard={"protect_decode": False},
                  no_protection={"protect_finish": False, "protect_decode": False})
_INITIALIZED = False


def motif(index, start_ms, background_prompt):
    prefix = f"m{index:03d}"
    return [
        dict(id=f"{prefix}:background_a", prompt=background_prompt,
             output=64, arrival_ms=start_ms),
        dict(id=f"{prefix}:background_b", prompt=background_prompt,
             output=64, arrival_ms=start_ms),
        dict(id=f"{prefix}:background_c", prompt=background_prompt,
             output=64, arrival_ms=start_ms),
        dict(id=f"{prefix}:target", prompt=480, output=16,
             arrival_ms=start_ms),
        dict(id=f"{prefix}:heavy_a", prompt=1200, output=64,
             arrival_ms=start_ms + HEAVY_OFFSET_MS),
        dict(id=f"{prefix}:heavy_b", prompt=1200, output=64,
             arrival_ms=start_ms + HEAVY_OFFSET_MS),
    ]


def requests(rate=None, background_prompt=400):
    if rate is None:
        return motif(0, 0., background_prompt)
    cycles = round(rate * HORIZON_MS / 6000.)
    if not math.isclose(cycles * 6000. / HORIZON_MS, rate, abs_tol=1e-9):
        raise ValueError("Requested QPS is not representable by complete motifs")
    period = HORIZON_MS / cycles
    result = []
    for index in range(cycles):
        result.extend(motif(index, (index + .5) * period, background_prompt))
    assert math.isclose(len(result) * 1000 / HORIZON_MS, rate, abs_tol=1e-9)
    assert result[-1]["arrival_ms"] < HORIZON_MS
    return result


class ProtectionObserver:
    def __init__(self):
        self.counts = Counter()
        self.events = []

    def inspect(self, row, scheduler):
        baseline = row["baseline"] or {}
        state = {r["rid"]: r for r in row["state"]}
        finish = {rid: quota for rid, quota in baseline.items()
                  if state[rid]["outputs"] == 0
                  and quota == state[rid]["tokens"] - state[rid]["computed"]}
        self.counts["rounds"] += 1
        self.counts["candidates_total"] += row["candidates"]
        self.counts["guard_rejections_total"] += row["rejected_guard"]
        self.counts["saving_rejections_total"] += row["rejected_saving"]
        self.counts["infeasible_rejections_total"] += row["rejected_infeasible"]
        self.counts["eligible_finish_rounds"] += bool(finish)
        self.counts["eligible_finish_instances"] += len(finish)
        for rid, quota in finish.items():
            if row["scheduled"].get(rid, 0) < quota:
                self.counts["selected_finish_delays"] += 1
                if rid.endswith(":target"):
                    self.counts["selected_target_delays"] += 1
        from vllm.v1.request import RequestStatus
        for r in row["state"]:
            if (r["rid"].endswith(":target") and r["running"]
                    and scheduler.requests[r["rid"]].status == RequestStatus.PREEMPTED):
                phase = "prefill" if r["outputs"] == 0 else "decode"
                self.counts[f"target_{phase}_preemptions"] += 1
        for number, candidate in enumerate(row["candidate_diagnostics"], 1):
            tokens = candidate["tokens"]
            if tokens is None:
                continue
            threats = [rid for rid, quota in finish.items() if tokens.get(rid, 0) < quota]
            decode_threats = [rid for rid, quota in baseline.items()
                              if state[rid]["outputs"] > 0
                              and state[rid]["tokens"] - state[rid]["computed"] == 1
                              and tokens.get(rid, 0) < quota]
            self.counts["finish_guard_checks"] += bool(finish)
            self.counts["finish_threatening_candidates"] += bool(threats)
            self.counts["finish_guard_rejections"] += bool(threats and not candidate["guard"])
            self.counts["finish_only_guard_rejections"] += bool(
                threats and not decode_threats and not candidate["guard"])
            self.counts["shared_finish_decode_guard_rejections"] += bool(
                threats and decode_threats and not candidate["guard"])
            self.counts["decode_only_guard_rejections"] += bool(
                decode_threats and not threats and not candidate["guard"])
            self.counts["finish_decision_relevant_candidates"] += bool(
                threats and candidate["saving"] >= row["threshold"]
                and row["baseline_growth_blocks"] > 0 and len(state) >= 2)
            if threats and len(self.events) < 12:
                target_id = next((rid for rid in threats if rid.endswith(":target")), threats[0])
                self.events.append(dict(round=row["round"], candidate=number,
                                        target_threat=any(r.endswith(":target") for r in threats),
                                        threatened_request=target_id,
                                        free_before=row["free_before"],
                                        threshold=row["threshold"],
                                        saving_blocks=candidate["saving"],
                                        guard_passed=candidate["guard"],
                                        base_target_tokens=baseline[target_id],
                                        candidate_target_tokens=tokens.get(target_id, 0),
                                        selected_target_tokens=row["scheduled"].get(target_id, 0)))


def replay(reqs, arm, *, capacity=CAPACITY, details=False):
    global _INITIALIZED
    if not _INITIALIZED:
        base.initialize()
        _INITIALIZED = True
    from research.www_evaluation_completion_20261003 import scheduler as scheduling
    observer = ProtectionObserver()
    original_make = base.ORIGINAL_MAKE

    def make(*args, **kwargs):
        scheduler, clock = original_make(*args, **kwargs)
        original_schedule = scheduler.schedule

        def schedule():
            decision = original_schedule()
            observer.inspect(scheduler.round_rows[-1], scheduler)
            return decision

        scheduler.schedule = schedule
        return scheduler, clock

    with patch.object(base.SIM, "make_scheduler", make), \
         patch.object(scheduling, "select", policy.select):
        result = base.SIM.simulate(
            reqs, arm="arrival", capacity=capacity, budget=1024, chunk=256,
            policy_options=MICRO_ARMS[arm], fitted_timing=base.TIMING,
            capture_probe=True, max_wall_s=180)
    records = result["probe_records"]
    gaps = result["probe_intervals_ms"]
    targets = [r for r in records if r["request_id"].endswith(":target")]
    others = [r for r in records if not r["request_id"].endswith(":target")]
    measured = [r for r in records if WARM_MS <= r["arrival_ms"] < HORIZON_MS]
    measured_gaps = [x for r in measured for x in gaps[r["request_id"]]]
    measured_targets = [r for r in measured if r["request_id"].endswith(":target")]
    measured_others = [r for r in measured if not r["request_id"].endswith(":target")]
    at = lambda t: sum(r["arrival_ms"] <= t < r["finish_ms"] for r in records)
    summary = dict(
        arm=arm, capacity_blocks=capacity, requests=len(reqs),
        trajectory_sha256=result["trajectory_sha256"],
        target_ttft_ms=targets[0]["ttft_ms"],
        target_ttft_mean_ms=float(np.mean([r["ttft_ms"] for r in targets])),
        other_p99_ttft_ms=float(np.percentile([r["ttft_ms"] for r in others], 99)),
        target_ttft_mean_measured_ms=(float(np.mean([r["ttft_ms"] for r in measured_targets]))
                                       if measured_targets else None),
        other_p99_ttft_measured_ms=(float(np.percentile(
            [r["ttft_ms"] for r in measured_others], 99)) if measured_others else None),
        preemptions=result["metrics"]["preemptions"],
        recomputed_tokens=result["metrics"]["recomputed_tokens"],
        p99_tbt_ms=float(np.percentile(measured_gaps, 99)) if measured_gaps else None,
        mean_ttft_ms=float(np.mean([r["ttft_ms"] for r in measured])) if measured else None,
        backlog_drift=at(HORIZON_MS)-at(MID_MS) if measured else None,
        counters=dict(observer.counts), events=observer.events)
    if details:
        summary["records"] = records
    return summary


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def micro():
    reqs = requests()
    design = dict(capacity_blocks=CAPACITY, control_capacity_blocks=512,
                  motif=reqs, background_prompts=BACKGROUND_PROMPTS,
                  micro_arms=MICRO_ARMS, capacity_arms=ARMS,
                  rates_qps=RATES, heavy_offset_ms=HEAVY_OFFSET_MS,
                  horizon_ms=HORIZON_MS, warm_ms=WARM_MS, mid_ms=MID_MS,
                  mean_ttft_limit_ms=THETA_MS, theta_grid_ms=THETA_GRID_MS,
                  p99_tbt_limit_ms=200,
                  backlog_limit="max(2, offered_qps)",
                  capacity_rule="all valid; mean of 3 motif conditions meets all limits; "
                                "ascending grid until first failure",
                  fitted_timing_sha256=base.digest(base.PREVIOUS / "timing_profile.json"),
                  policy_sha256=base.digest(Path(policy.__file__)),
                  runner_sha256=base.digest(Path(__file__)),
                  simulator_only=True)
    save(OUT / "design.json", design)
    rows = {arm: replay(reqs, arm, details=True) for arm in MICRO_ARMS}
    control = {arm: replay(reqs, arm, capacity=512) for arm in ARMS}
    assert rows["full"]["trajectory_sha256"] == rows["no_decode_guard"]["trajectory_sha256"]
    assert rows["no_finish_guard"]["trajectory_sha256"] == rows["no_protection"]["trajectory_sha256"]
    rows["wide_kv_control"] = control
    save(OUT / "micro.json", rows)
    for arm, row in rows.items():
        print(arm, json.dumps({k: v for k, v in row.items() if k != "records"}))


def capacity_task(task):
    background_prompt, rate, arm = task
    reqs = requests(rate, background_prompt)
    row = replay(reqs, arm)
    row.pop("events")
    row.update(background_prompt=background_prompt,
               offered_qps=len(reqs) * 1000 / HORIZON_MS)
    return row


def evaluate(rows, rate, theta_ms):
    if len(rows) != len(BACKGROUND_PROMPTS):
        raise ValueError("incomplete paired scenario grid")
    metrics = {key: float(np.mean([r[key] for r in rows]))
               for key in ("mean_ttft_ms", "p99_tbt_ms", "backlog_drift")}
    reasons = []
    if metrics["mean_ttft_ms"] > theta_ms:
        reasons.append("mean_ttft")
    if metrics["p99_tbt_ms"] > 200:
        reasons.append("p99_tbt")
    if metrics["backlog_drift"] > max(2, rate):
        reasons.append("backlog")
    return dict(**metrics, passes=not reasons, failure_causes=reasons)


def scan(workers):
    if not (OUT / "micro.json").exists():
        micro()
    tasks = [(prompt, rate, arm) for prompt in BACKGROUND_PROMPTS
             for rate in RATES for arm in ARMS]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(capacity_task, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)}: {row['background_prompt']} "
                  f"{row['offered_qps']:g} {row['arm']}", flush=True)
    rows.sort(key=lambda r: (r["offered_qps"], r["background_prompt"], r["arm"]))
    save(OUT / "rates.json", rows)
    groups = []
    for theta_ms in THETA_GRID_MS:
        for arm in ARMS:
            for rate in RATES:
                selected = [r for r in rows if r["arm"] == arm and r["offered_qps"] == rate]
                groups.append(dict(theta_ms=theta_ms, arm=arm, offered_qps=rate,
                                   **evaluate(selected, rate, theta_ms)))
    capacity = {}
    for theta_ms in THETA_GRID_MS:
        capacity[str(theta_ms)] = {}
        for arm in ARMS:
            passing = 0.
            first_fail = None
            for group in (g for g in groups if g["theta_ms"] == theta_ms
                          and g["arm"] == arm):
                if not group["passes"]:
                    first_fail = group
                    break
                passing = group["offered_qps"]
            capacity[str(theta_ms)][arm] = dict(last_pass_qps=passing,
                                                 first_fail=first_fail)
    save(OUT / "capacity.json", dict(groups=groups, capacity=capacity))
    print(json.dumps(capacity, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.micro:
        micro()
    elif args.scan:
        scan(args.workers)
