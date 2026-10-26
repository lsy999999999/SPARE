"""WWW baseline replay: frozen basic+A on the native V1 CPU scheduler.

No model, CUDA execution, or future-length input to the policy. Timing remains
explicitly synthetic until an independently validated profile is supplied.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.v1_broad12h_20260925.policy import formula
from research.www_evaluation_completion_20261003.scheduler import EvaluationScheduler as FormulaScheduler
from research.v1_broad12h_20260925.policy import ArrivalRate
from research.v1_budget_20260922.scheduler import BudgetScheduler
from research.v1_paper_comparison_20260922.scheduler import ComparisonScheduler
from simulator_core.timing import ResourceTiming, TimingBatch
from simulator_core.types import percentile

ARMS = ("arrival", "basic", "srp", "native", "sarathi", "past_future",
        "ssjf_oracle", "mc_sf_oracle", "trail_oracle", "tie_oracle_proxy",
        "uniboost_static", "apt_serve_kv_rule", "sola_rule")
ORACLE_ARMS = ("ssjf_oracle", "mc_sf_oracle", "trail_oracle",
               "tie_oracle_proxy")
PAPER_ARMS = ("sarathi", "past_future") + ORACLE_ARMS + (
    "uniboost_static", "apt_serve_kv_rule", "sola_rule")
ADAPTATION = {
    "arrival": ("current_method", ()),
    "basic": ("ablation", ()),
    "srp": ("native_rule", ()),
    "native": ("native_system", ()),
    "sarathi": ("paper_rule_adapted", ("stall-free batch invariant",)),
    "past_future": ("paper_rule_adapted", ("original serving stack",)),
    "ssjf_oracle": ("paper_oracle", ("BERT proxy predictor",)),
    "mc_sf_oracle": ("paper_oracle", ("unknown-length deployment",)),
    "trail_oracle": ("paper_oracle", ("embedding remaining-length predictor", "KV swap")),
    "tie_oracle_proxy": ("paper_oracle_proxy", ("DeBERTa distribution predictor", "async prediction latency")),
    "uniboost_static": ("paper_rule_adapted", ("adaptive gamma controller", "KV swap; recompute used")),
    "apt_serve_kv_rule": ("paper_rule_adapted", ("hybrid hidden cache", "cross-phase batch selection")),
    "sola_rule": ("paper_rule_adapted", ("learned hardware cost model", "cross-phase batch selection")),
}
DEFAULT_OUTPUT = ROOT / "writing/www/experiments/native_simulator_20260929"
TIMING_PARAMETERS = dict(launch_ms=1., weight_read_ms=5., prefill_ms_per_token=.08,
    prefill_context_ms_per_token2=.000003, decode_ms_per_request=.03,
    decode_context_ms_per_token=.00001)
SOURCE_FILES = [
    'research/www_evaluation_completion_20261003/simulate.py',
    'research/www_evaluation_completion_20261003/scheduler.py',
    'research/www_evaluation_completion_20261003/policy.py',
    "writing/www/code/run_simulator.py", "simulator_core/timing.py",
    "research/v1_broad12h_20260925/policy.py",
    "research/v1_broad12h_20260925/scheduler.py",
    "research/v1_saving_threshold_20260924/policy.py",
    "research/v1_budget_20260922/policy.py",
    "research/v1_budget_20260922/scheduler.py",
    "research/v1_paper_comparison_20260922/scheduler.py",
    "research/v1_paper_comparison_20260922/rules.py",
]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_scheduler(arm="arrival", blocks=768, budget=1024, chunk=256,
                   concurrency=32, max_model_len=8176, history=(), seed=0,
                   oracle_lengths=None, policy_options=None):
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    if arm not in ARMS or blocks < 2 or not 0 < chunk <= budget <= 2048:
        raise ValueError("invalid policy/capacity/budget/chunk")
    cls = FormulaScheduler if arm in ("arrival", "basic") else (
        ComparisonScheduler if arm in PAPER_ARMS
        else BudgetScheduler)
    config = NS(scheduler_config=NS(max_num_seqs=64, max_num_batched_tokens=2048,
        max_model_len=max_model_len, policy="fcfs", long_prefill_token_threshold=chunk,
        chunked_prefill_enabled=True),
        cache_config=NS(num_gpu_blocks=blocks, block_size=16,
            enable_prefix_caching=False, prefix_caching_hash_algo="builtin"),
        lora_config=None, kv_events_config=None, kv_transfer_config=None,
        speculative_config=None, model_config=None,
        parallel_config=NS(data_parallel_rank=0, pipeline_parallel_size=1))
    kv = KVCacheConfig(blocks, [], [KVCacheGroupSpec(["layer"],
        FullAttentionSpec(16, 1, 16, torch.float16, False))])
    # Decoder-only CPU harness: no model weights or encoder workload is loaded.
    with patch("vllm.v1.core.sched.scheduler.compute_encoder_budget", return_value=(0, 0)):
        scheduler = cls(config, kv, NS(grammar_bitmask=lambda *args: None))
    clock = [0.]
    scheduler.arrival_clock = lambda: clock[0] / 1000
    if arm in ("arrival", "basic"):
        scheduler.configure("threshold", budget, chunk, concurrency,
                            threshold=formula(1 if arm == "arrival" else 0))
    elif arm in PAPER_ARMS:
        scheduler.history_context = history
        scheduler.seed_context = seed
        scheduler.oracle_lengths_context = oracle_lengths or {}
        method = {"sarathi": "sarathi_v1", "past_future": "past_future_v1"}.get(
            arm, arm)
        scheduler.configure(method, budget, chunk, concurrency)
    else:
        scheduler.configure(arm, budget, chunk, concurrency)
    if arm in ('arrival', 'basic'):
        scheduler.evaluation_options = dict(policy_options or {})
        scheduler.arrival_estimator = ArrivalRate(scheduler.evaluation_options.get('H', 1.0))
    return scheduler, clock


def distribution(values, name):
    values = list(values)
    return {name + "_" + label: (fun(values) if values else None)
            for label, fun in [("mean", statistics.fmean), ("p50", lambda x: percentile(x, 50)),
                              ("p90", lambda x: percentile(x, 90)),
                              ("p95", lambda x: percentile(x, 95)),
                              ("p99", lambda x: percentile(x, 99)), ("max", max)]}


def simulate(requests, *, arm="arrival", capacity=768, budget=1024, chunk=256,
             profile="resource", history=(), seed=0, output=None,
             max_rounds=500000, max_wall_s=900, policy_options=None, fitted_timing=None,
             full_trace=False, capture_probe=False, stop_at_ms=None,
             probe_times_ms=()):
    from vllm import SamplingParams
    from vllm.v1.request import Request, RequestStatus
    if profile not in ("resource", "resource_staircase"):
        raise ValueError(profile)
    # Stable sorting preserves original submission order for simultaneous arrivals.
    requests = sorted(requests, key=lambda r: r["arrival_ms"])
    if not requests or len({r["id"] for r in requests}) != len(requests):
        raise ValueError("requests must be nonempty with unique IDs")
    if stop_at_ms is not None and (stop_at_ms <= 0 or
                                   any(t < 0 or t > stop_at_ms for t in probe_times_ms)):
        raise ValueError("invalid stop time or probe checkpoint")
    for r in requests:
        if not (r["prompt"] > 0 and r["output"] > 0 and r["prompt"] + r["output"] < 8176
                and math.isfinite(r["arrival_ms"]) and r["arrival_ms"] >= 0):
            raise ValueError("invalid request length or arrival")
        if math.ceil((r["prompt"] + r["output"] - 1) / 16) > capacity - 1:
            raise ValueError("individual realized request cannot fit usable KV capacity")
    oracle_lengths = ({r["id"]: r["output"] for r in requests}
                      if arm in ORACLE_ARMS else None)
    scheduler, clock = make_scheduler(arm, capacity, budget, chunk, history=history,
                                      seed=seed, oracle_lengths=oracle_lengths, policy_options=policy_options)
    timing = ResourceTiming(dict(measurement_scope="synthetic-uncalibrated", parameters=TIMING_PARAMETERS))
    targets = {r["id"]: r["output"] for r in requests}  # Used ONLY by executor to stop generation.
    arrivals = {r["id"]: r["arrival_ms"] for r in requests}
    token_times = {r["id"]: [] for r in requests}
    first_scheduled = {}
    submitted = {}
    finish = {}
    rounds = cursor = preemptions = recomputed = executed = interventions = peak = 0
    kv_area = 0.
    wall = time.monotonic()
    schedule_costs = []
    trajectory = hashlib.sha256()
    output = Path(output) if output else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    log = gzip.open(output / "iterations.jsonl.gz", "wt", compresslevel=1) if output else None
    try:
        while cursor < len(requests) or scheduler.requests:
            if stop_at_ms is not None and clock[0] >= stop_at_ms:
                break
            if rounds >= max_rounds or time.monotonic() - wall > max_wall_s:
                raise RuntimeError("simulation limit; no completed result will be reported")
            if not scheduler.requests:
                clock[0] = max(clock[0], requests[cursor]["arrival_ms"])
            while cursor < len(requests) and requests[cursor]["arrival_ms"] <= clock[0]:
                r = requests[cursor]
                # A common context-bound output cap, never the realized output length.
                scheduler.add_request(Request(r["id"], [100] * r["prompt"], None, None, None,
                    SamplingParams(max_tokens=8176-r["prompt"]), None, None,
                    arrival_time=r["arrival_ms"] / 1000))
                submitted[r["id"]] = clock[0]
                cursor += 1
            started_ms = clock[0]
            decision = scheduler.schedule()
            row = dict(scheduler.round_rows[-1])
            pre, contexts, dec = [], [], []
            for allocation in row["allocations"]:
                rid, count, before = allocation["id"], allocation["tokens"], allocation["before"]
                first_scheduled.setdefault(rid, clock[0])
                if allocation["outputs"] > 0 and count == 1 and before + 1 == scheduler.requests[rid].num_tokens:
                    dec.append(before)
                else:
                    pre.append(count)
                    contexts.append(before)
            if not decision.num_scheduled_tokens:
                raise RuntimeError("empty schedule with pending work; trace/policy cannot progress")
            batch = TimingBatch(tuple(pre), tuple(contexts), tuple(dec))
            duration, _ = timing.estimate(batch) if fitted_timing is None else fitted_timing.estimate(batch)
            if profile == "resource_staircase" and fitted_timing is None:
                duration += 5 * math.ceil(sum(decision.num_scheduled_tokens.values()) / 256)
            clock[0] += duration
            row['pending_first_tokens'] = sum(not ts for rid, ts in token_times.items() if rid in submitted)
            row['unfinished_requests'] = len(scheduler.requests)
            done = []
            for rid in decision.num_scheduled_tokens:
                request = scheduler.requests[rid]
                if request.num_computed_tokens == request.num_tokens:
                    request.append_output_token_ids(100)
                    token_times[rid].append(clock[0])
                if len(request.output_token_ids) == targets[rid]:
                    finish[rid] = clock[0]
                    done.append(rid)
            occupied = capacity - 1 - row["free_after"]
            assert 0 <= occupied <= capacity - 1
            peak = max(peak, occupied)
            kv_area += occupied * duration
            scheduler.finish_requests(done, RequestStatus.FINISHED_LENGTH_CAPPED)
            preemptions += row["preemptions"]
            recomputed += row["recomputed_tokens"]
            executed += sum(decision.num_scheduled_tokens.values())
            interventions += row["intervened"]
            schedule_costs.append(row["schedule_ms"])
            rounds += 1
            trajectory.update(json.dumps([row["scheduled"], row["free_before"], row["free_after"], clock[0]], sort_keys=True).encode())
            if log:
                row.update(round=rounds-1, simulated_start_ms=started_ms,
                           simulated_finish_ms=clock[0], simulated_step_ms=duration)
                if not full_trace:
                    for field in ('state', 'state_fingerprint', 'candidate_diagnostics', 'allocations', 'scheduled', 'baseline', 'consecutive_reductions', 'arrival_events'):
                        row.pop(field, None)
                log.write(json.dumps(row, allow_nan=False) + '\n')
            if len(scheduler.round_rows) > 16:
                scheduler.round_rows = scheduler.round_rows[-8:]
    finally:
        if log:
            log.close()
    if stop_at_ms is not None:
        checkpoints = sorted(set((*probe_times_ms, stop_at_ms)))
        windows = []
        for at in checkpoints:
            arrived = [rid for rid, t in arrivals.items() if t <= at]
            completed = [rid for rid, t in finish.items() if t <= at]
            waiting = sum(not token_times[rid] or token_times[rid][0] > at
                          for rid in arrived)
            windows.append(dict(at_ms=at, arrivals=len(arrived),
                                completed=len(completed),
                                completed_output_tokens=sum(targets[rid] for rid in completed),
                                emitted_output_tokens=sum(bisect_right(ts, at)
                                                          for ts in token_times.values()),
                                outstanding=len(arrived)-len(completed),
                                waiting_first_token=waiting))
        return dict(arm=arm, capacity_configured_blocks=capacity, budget=budget,
                    chunk=chunk, complete=False, stopped_at_ms=stop_at_ms,
                    elapsed_simulated_ms=clock[0], cpu_wall_s=time.monotonic()-wall,
                    rounds=rounds, preemptions=preemptions,
                    recomputed_tokens=recomputed, pressure_windows=windows,
                    trajectory_sha256=trajectory.hexdigest())
    assert len(finish) == len(requests)
    assert all(len(token_times[rid]) == count for rid, count in targets.items())
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == capacity - 1
    assert executed - sum(r["prompt"] + r["output"] - 1 for r in requests) == recomputed
    records, gaps = [], []
    for r in requests:
        rid, ts = r["id"], token_times[r["id"]]
        itls = [b-a for a, b in zip(ts, ts[1:])]
        gaps.extend(itls)
        records.append(dict(request_id=rid, arrival_ms=arrivals[rid], submitted_ms=submitted[rid],
            prompt_tokens=r["prompt"], output_tokens=r["output"], first_token_ms=ts[0], finish_ms=finish[rid],
            ttft_ms=ts[0]-arrivals[rid], e2e_ms=finish[rid]-arrivals[rid],
            scheduling_delay_ms=first_scheduled[rid]-arrivals[rid],
            tpot_ms=(ts[-1]-ts[0])/(len(ts)-1) if len(ts)>1 else None,
            max_itl_ms=max(itls) if itls else None,
            normalized_e2e_ms=(finish[rid]-arrivals[rid])/len(ts)))
    elapsed = clock[0] - min(arrivals.values())
    metrics = dict(completed=len(records), requests=len(requests), rounds=rounds,
        preemptions=preemptions, recomputed_tokens=recomputed, executed_tokens=executed,
        interventions=interventions, intervention_fraction=interventions/rounds,
        peak_kv_blocks=peak, peak_kv_util=peak/(capacity-1),
        time_weighted_kv_util=kv_area/elapsed/(capacity-1), elapsed_simulated_ms=elapsed,
        request_throughput_rps=len(records)*1000/elapsed,
        output_throughput_tps=sum(targets.values())*1000/elapsed,
        ttft_waiting_area_ms_requests=sum(r["ttft_ms"] for r in records),
        cpu_wall_s=time.monotonic()-wall)
    for field in ("ttft_ms", "e2e_ms", "scheduling_delay_ms", "tpot_ms", "max_itl_ms", "normalized_e2e_ms"):
        metrics.update(distribution((r[field] for r in records if r[field] is not None), field))
    metrics.update(distribution(gaps, "token_itl_ms"))
    metrics.update(distribution(schedule_costs, "cpu_schedule_including_audit_ms"))
    metrics["sarathi_first_zero_token_tbt_p99_ms"] = percentile(gaps + [0.] * len(records), 99)
    for ttft_slo in (1000, 5000, 10000):
        for itl_slo in (50, 100, 200):
            count = sum(r["ttft_ms"] <= ttft_slo and (r["max_itl_ms"] or 0) <= itl_slo for r in records)
            metrics[f"slo_ttft{ttft_slo}_maxitl{itl_slo}_attainment"] = count/len(records)
            metrics[f"slo_ttft{ttft_slo}_maxitl{itl_slo}_goodput_rps"] = count*1000/elapsed
    result = dict(arm=arm, capacity_configured_blocks=capacity, capacity_usable_blocks=capacity-1,
        budget=budget, chunk=chunk, concurrency=32, block_tokens=16, profile=profile, policy_options=policy_options,
        timing_parameters=TIMING_PARAMETERS, seed=seed, complete=True, metrics=metrics,
        trajectory_sha256=trajectory.hexdigest(), performance_validated=False,
        timing_claim="synthetic-uncalibrated; not GPU performance",
        adaptation_level=ADAPTATION[arm][0],
        uses_realized_output_lengths=arm in ORACLE_ARMS,
        deployable=arm not in ORACLE_ARMS,
        paper_mechanism_omitted=list(ADAPTATION[arm][1]),
        scheduler_overhead_injected=False,
        source_sha256={p:digest(ROOT/p) for p in SOURCE_FILES})
    if capture_probe:
        result['probe_records'] = records
        result['probe_intervals_ms'] = {
            rid: [b-a for a, b in zip(ts, ts[1:])]
            for rid, ts in token_times.items()}
    if output:
        csv_write(output / "requests.csv", records)
        write(output / "result.json", result)
    return result


def run_case_arm(task):
    case, arm, output_dir = task
    inputs = ROOT / case["input"]
    history_path = ROOT / case["history"]
    history_rows = [json.loads(line) for line in history_path.read_text().splitlines()
                    if line.strip()]
    history = [int(row["actual_output_tokens"]) for row in history_rows]
    requests = read(inputs)
    overlap = {str(row.get("source_request_id", row["id"])) for row in requests} & {
        str(row["request_id"]) for row in history_rows}
    if overlap:
        raise ValueError(f"evaluation/history request overlap: {overlap}")
    name = case["name"] + "_" + arm
    result = simulate(requests, arm=arm, capacity=case["capacity"],
        budget=case.get("budget", 1024), chunk=case.get("chunk", 256),
        profile=case.get("profile", "resource"), history=history,
        seed=case.get("seed", 0), output=Path(output_dir) / name)
    result.update(name=name, input=case["input"], input_sha256=digest(inputs),
                  history=case["history"], history_sha256=digest(history_path))
    write(Path(output_dir) / name / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT/"writing/www/experiments/suite.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    suite = read(args.suite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arms = suite.get("arms", ARMS)
    tasks = [(case, arm, str(args.output_dir)) for case in suite["cases"] for arm in arms]
    results = []
    if args.jobs <= 1:
        iterator = map(run_case_arm, tasks)
        for result in iterator:
            results.append(result)
            write(args.output_dir/"status.json", dict(completed=len(results), expected=len(tasks)))
            print(result["name"], "TTFT", round(result["metrics"]["ttft_ms_mean"], 3), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_case_arm, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                write(args.output_dir/"status.json", dict(completed=len(results), expected=len(tasks)))
                print(result["name"], "TTFT", round(result["metrics"]["ttft_ms_mean"], 3), flush=True)
    order = {(case["name"], arm): index for index, (case, arm, _) in enumerate(tasks)}
    results.sort(key=lambda result: order[(result["name"][:-len(result["arm"])-1], result["arm"])])
    write(args.output_dir/"summary.json", results)
    csv_write(args.output_dir/"metrics.csv", [dict(name=r["name"], arm=r["arm"], **r["metrics"]) for r in results])


if __name__ == "__main__":
    main()
