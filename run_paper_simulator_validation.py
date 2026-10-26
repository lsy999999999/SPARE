#!/usr/bin/env python3
"""Run paper policies in the existing engine and compare matched GPU measurements."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from simulator_core import LLMSingleNodeSimulator, SimulatorConfig, load_trace
from simulator_core.paper_schedulers import PaperScheduler, TIEPrediction
from simulator_core.types import percentile


RECOVERY_V2 = Path(__file__).resolve().parent / "research/kv_risk_feasibility_20260909/recovery_cost_v2"
if str(RECOVERY_V2) not in sys.path:
    sys.path.insert(0, str(RECOVERY_V2))

from survival_conditioned_reservation import StaticTailBoundedReservationGate


POLICY_PROVENANCE = {
    "fcfs": "Orca OSDI 2022 / vLLM SOSP 2023",
    "orca-max": "vLLM SOSP 2023 baseline",
    "orca-pow2": "vLLM SOSP 2023 offline baseline",
    "orca-oracle": "vLLM SOSP 2023 oracle baseline",
    "s3": "S3 NeurIPS 2023",
    "ltr": "LTR NeurIPS 2024",
    "ltr-oracle": "LTR NeurIPS 2024 derived diagnostic",
    "mlfq": "generic MLFQ; LTR NeurIPS 2024 baseline",
    "sarathi": "Sarathi-Serve OSDI 2024",
    "at-bb": "Active-Tail Budgeted Backfilling",
    "conservative": "Past-Future ASPLOS 2025 conservative baseline",
    "past-future": "Past-Future ASPLOS 2025 rule adaptation",
    "tie": "TIE arXiv preprint",
}

IMPLEMENTATION_FILES = (
    "simulator_core/engine.py",
    "simulator_core/types.py",
    "simulator_core/workload.py",
    "simulator_core/paper_schedulers.py",
    "simulator_core/timing.py",
    "research/kv_risk_feasibility_20260909/recovery_cost_v2/run_probe.py",
    "research/kv_risk_feasibility_20260909/recovery_cost_v2/bounded_reservation_validation.py",
    "research/kv_risk_feasibility_20260909/recovery_cost_v2/survival_conditioned_reservation.py",
    "run_paper_simulator_validation.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def error_metrics(simulated: list[float], measured: list[float]) -> dict:
    if not simulated or len(simulated) != len(measured):
        raise ValueError("reference and simulation must contain the same nonzero number of samples")
    if any(not math.isfinite(v) or v < 0 for v in simulated + measured):
        raise ValueError("latencies must be finite and nonnegative")
    differences = [a - b for a, b in zip(simulated, measured)]
    percentage = [100 * abs(a - b) / b for a, b in zip(simulated, measured) if b > 0]
    sse = sum(d * d for d in differences)
    center = statistics.fmean(measured)
    sst = sum((b - center) ** 2 for b in measured)
    return {
        "n": len(measured), "bias_ms": statistics.fmean(differences),
        "mae_ms": statistics.fmean(abs(d) for d in differences),
        "rmse_ms": math.sqrt(sse / len(measured)),
        "mape_pct": statistics.fmean(percentage) if percentage else None,
        "mape_n": len(percentage), "r2": 1 - sse / sst if sst > 0 else None,
    }


def compare_reference(path: Path | None, signature: str, rows: list[dict], completed: bool) -> dict:
    if path is None:
        return {"status": "missing_matched_gpu_measurements", "accuracy": None}
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("source_kind") != "gpu-measured" or not reference.get("source"):
        raise ValueError("reference needs source_kind=gpu-measured and a measurement source")
    if reference.get("measurement_role") != "validation":
        raise ValueError("accuracy evaluation requires validation measurements, not calibration data")
    if reference.get("experiment_signature") != signature:
        raise ValueError("reference experiment signature does not match policy/trace/config/seed/predictor")
    if not completed:
        return {"status": "incomplete_simulation", "accuracy": None}
    records = reference["requests"]
    measured = {int(row["request_id"]): row for row in records}
    if len(measured) != len(records) or set(measured) != {r["request_id"] for r in rows}:
        raise ValueError("reference must have exactly one record for every simulated request")
    if set(reference.get("calibration_request_ids", [])) & set(measured):
        raise ValueError("calibration and validation request IDs overlap")
    metrics = {}
    deltas = []
    for metric in ("ttft_ms", "e2e_ms", "tbt_max_ms"):
        matched = []
        for row in rows:
            actual = measured[row["request_id"]].get(metric)
            predicted = row[metric]
            if actual is None or predicted is None:
                if metric != "tbt_max_ms" or actual != predicted:
                    raise ValueError(f"missing or inconsistent {metric} in reference")
                continue
            actual, predicted = float(actual), float(predicted)
            matched.append((predicted, actual))
            deltas.append({"request_id": row["request_id"], "metric": metric,
                           "simulated_ms": predicted, "measured_ms": actual,
                           "delta_ms": predicted - actual,
                           "relative_error_pct": 100 * (predicted - actual) / actual if actual else None})
        metrics[metric] = error_metrics([x for x, _ in matched], [y for _, y in matched]) if matched else None
    return {"status": "matched_gpu_comparison", "source": reference["source"],
            "accuracy": metrics, "request_deltas": deltas}


def request_rows(engine: LLMSingleNodeSimulator, scheduler: PaperScheduler) -> list[dict]:
    timestamps = {r.request_id: [] for r in engine.requests}
    if engine.config.execution_mode == "scheduled-tokens":
        # One-token decodes give actual consecutive-token intervals, including prefill stalls.
        for step in scheduler.steps:
            token_ids = set(str(step["decode_progress_request_ids"]).split(","))
            token_ids.update(str(step.get("first_token_request_ids", "")).split(","))
            for request_id in token_ids:
                if request_id:
                    timestamps[int(request_id)].append(float(step["current_time_ms"]))
    rows = []
    for request in engine.requests:
        times = timestamps[request.request_id]
        gaps = [b - a for a, b in zip(times, times[1:])]
        rows.append({
            "request_id": request.request_id, "arrival_ms": request.arrival_ms,
            "prompt_tokens": request.prompt_tokens, "output_tokens": request.actual_output_tokens,
            "ttft_ms": request.ttft_ms, "e2e_ms": request.e2e_ms,
            "scheduling_delay_ms": (request.prefill_started_at_ms - request.arrival_ms
                                    if request.prefill_started_at_ms is not None else None),
            "per_token_latency_ms": request.e2e_ms / request.actual_output_tokens if request.is_complete else None,
            "tbt_max_ms": max(gaps) if gaps else None,
            "tbt_p95_ms": percentile(gaps, 95) if gaps else None,
            "evictions": request.eviction_count, "complete": request.is_complete,
        })
    return rows


def paper_token_tbt_values(scheduler: PaperScheduler) -> list[float]:
    """Match Sarathi's per-decode-token metric, including zero for each first token."""
    timestamps: dict[int, list[float]] = {}
    for step in scheduler.steps:
        token_ids = set(str(step["decode_progress_request_ids"]).split(","))
        token_ids.update(str(step.get("first_token_request_ids", "")).split(","))
        for request_id in token_ids:
            if request_id:
                timestamps.setdefault(int(request_id), []).append(float(step["current_time_ms"]))
    values = []
    for times in timestamps.values():
        if times:
            values.append(0.0)
            values.extend(b - a for a, b in zip(times, times[1:]))
    return values


def load_predictions(path: Path | None, seed: int) -> dict[int, TIEPrediction]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("source"):
        raise ValueError("TIE prediction file must record its predictor source")
    predictions = {}
    for row in data["requests"]:
        request_id = int(row["request_id"])
        if request_id in predictions:
            raise ValueError("duplicate TIE prediction ID")
        delay = float(row.get("ready_after_ms", 0))
        if "logt_mu" in row:
            prediction = TIEPrediction.from_log_t(float(row["logt_mu"]), float(row["logt_sigma"]),
                                                int(row["max_tokens"]), seed=seed + request_id,
                                                ready_after_ms=delay)
        else:
            prediction = TIEPrediction(float(row["censored_mean"]), float(row["cvar90"]), delay)
        predictions[request_id] = prediction
    return predictions


def run_case(spec: dict, root: Path, output: Path) -> dict:
    seed = int(spec.get("seed", 0))
    trace_path = (root / spec["trace"]).resolve()
    requests = load_trace(trace_path, seed=seed)
    if not requests or len({r.request_id for r in requests}) != len(requests):
        raise ValueError("trace must be nonempty with unique request IDs")
    if any(r.actual_output_tokens <= 0 for r in requests):
        raise ValueError("matched replay requires fixed realized output lengths for every request")
    if any(not math.isfinite(r.arrival_ms) or r.arrival_ms < 0 for r in requests):
        raise ValueError("arrival timestamps must be finite and nonnegative")
    max_output_tokens = int(spec.get("max_output_tokens", 4096))
    uniform_max_output_tokens = bool(spec.get("uniform_max_output_tokens", False))
    if uniform_max_output_tokens:
        for request in requests:
            request.max_new_tokens = max_output_tokens
            request.actual_output_tokens = min(request.actual_output_tokens, max_output_tokens)
    arrival_scale = float(spec.get("arrival_scale", 1.0))
    if not math.isfinite(arrival_scale) or arrival_scale <= 0:
        raise ValueError("arrival_scale must be positive and finite")
    poisson_qps = spec.get("poisson_qps")
    if poisson_qps is not None:
        poisson_qps = float(poisson_qps)
        if not math.isfinite(poisson_qps) or poisson_qps <= 0 or arrival_scale != 1:
            raise ValueError("poisson_qps must be positive and cannot be combined with arrival_scale")
        arrival_rng = random.Random(int(spec.get("arrival_seed", seed)))
        arrival_ms = 0.0
        for request in requests:
            request.arrival_ms = arrival_ms
            interval_s = min(arrival_rng.expovariate(poisson_qps), 3.0 / poisson_qps)
            arrival_ms += interval_s * 1000
    else:
        first = min(r.arrival_ms for r in requests)
        for request in requests:
            request.arrival_ms = (request.arrival_ms - first) / arrival_scale
    config = SimulatorConfig(**spec.get("simulator", {}))
    if config.timing_profile is not None and any(
        config.timing_profile[key] != spec.get(key) for key in ("model", "hardware")
    ):
        raise ValueError("timing profile model/hardware must match the experiment")
    if config.simulation_duration_ms != 0 or config.force_output_length_sampling:
        raise ValueError("paper replay must drain fixed outputs without resampling")
    if config.time_unit != "ms":
        raise ValueError("latency comparison requires time_unit=ms")
    history_path = (root / spec["history_trace"]).resolve() if spec.get("history_trace") else None
    if history_path == trace_path:
        raise ValueError("Past-Future history must precede and differ from the evaluation trace")
    history = [r.actual_output_tokens for r in load_trace(history_path, seed=seed)] if history_path else []
    if uniform_max_output_tokens:
        history = [min(length, max_output_tokens) for length in history]
    prediction_path = (root / spec["predictions"]).resolve() if spec.get("predictions") else None
    predictions = load_predictions(prediction_path, seed)
    if spec["policy"] == "tie" and set(predictions) != {r.request_id for r in requests}:
        raise ValueError("TIE requires one external prediction per trace request")
    max_batch_size = int(spec.get("max_batch_size", 16))
    if spec["policy"] == "at-bb":
        if not history:
            raise ValueError("AT-BB requires a train-only history_trace")
        scheduler = StaticTailBoundedReservationGate(
            config,
            "shortest_safe_output_gate",
            float(spec.get("atbb_overflow_threshold", 0.0)),
            history,
            seed=seed,
            sample_count=int(spec.get("atbb_sample_count", 64)),
            rank_max_wait_ms=float(spec.get("atbb_age_guard_ms", 1_000.0)),
            use_request_distribution=True,
            max_batch_size=max_batch_size,
            delay_budget_ms=float(spec.get("atbb_delay_budget_ms", 100.0)),
            tail_floor=float(spec.get("atbb_tail_floor", 0.75)),
        )
    else:
        scheduler = PaperScheduler(config, spec["policy"], max_batch_size=max_batch_size,
                                   max_output_tokens=max_output_tokens, history=history,
                                   seed=seed, predictions=predictions,
                                   s3_bucket_count=int(spec.get("s3_bucket_count", 10)),
                                   ltr_starvation_threshold=int(spec.get("ltr_starvation_threshold", 100)),
                                   ltr_priority_quantum=int(spec.get("ltr_priority_quantum", 10)),
                                   mlfq_base_quantum_ms=float(spec.get("mlfq_base_quantum_ms", 16_000)),
                                   mlfq_quantum_growth=float(spec.get("mlfq_quantum_growth", 2)))
    criteria = spec.get("capacity_criteria")
    if criteria is not None:
        criteria = {"p99_tbt_slo_ms": float(criteria["p99_tbt_slo_ms"]),
                    "p50_scheduling_delay_slo_ms": float(criteria.get("p50_scheduling_delay_slo_ms", 2000))}
        if any(not math.isfinite(v) or v <= 0 for v in criteria.values()):
            raise ValueError("capacity criteria must be finite and positive")
        if config.execution_mode != "scheduled-tokens":
            raise ValueError("capacity criteria require scheduled-tokens execution")
    identity = {"policy": spec["policy"], "trace_sha256": digest(trace_path), "arrival_scale": arrival_scale,
                "poisson_qps": poisson_qps, "arrival_seed": int(spec.get("arrival_seed", seed)),
                "seed": seed, "simulator": asdict(config), "max_batch_size": scheduler.max_batch_size,
                "max_output_tokens": max_output_tokens,
                "history_sha256": digest(history_path) if history_path else None,
                "predictions_sha256": digest(prediction_path) if prediction_path else None,
                "model": spec.get("model", "unprofiled"), "hardware": spec.get("hardware", "unprofiled"),
                "first_token_convention": config.first_token_mode,
                "capacity_criteria": criteria,
                "comparison_group": spec.get("comparison_group"),
                "baseline_name": spec.get("baseline_name"),
                "publication_provenance": POLICY_PROVENANCE[spec["policy"]],
                "uses_hidden_output_for_scheduling": spec["policy"] in PaperScheduler.ORACLE_POLICIES,
                "uniform_max_output_tokens": uniform_max_output_tokens,
                "s3_bucket_count": getattr(scheduler, "s3_bucket_count", int(spec.get("s3_bucket_count", 10))),
                "ltr_starvation_threshold": getattr(scheduler, "ltr_starvation_threshold", int(spec.get("ltr_starvation_threshold", 100))),
                "ltr_priority_quantum": getattr(scheduler, "ltr_priority_quantum", int(spec.get("ltr_priority_quantum", 10))),
                "mlfq_base_quantum_ms": getattr(scheduler, "mlfq_base_quantum_ms", float(spec.get("mlfq_base_quantum_ms", 16_000))),
                "mlfq_quantum_growth": getattr(scheduler, "mlfq_quantum_growth", float(spec.get("mlfq_quantum_growth", 2))),
                "atbb_tail_floor": float(spec.get("atbb_tail_floor", 0.75)) if spec["policy"] == "at-bb" else None,
                "atbb_delay_budget_ms": float(spec.get("atbb_delay_budget_ms", 100.0)) if spec["policy"] == "at-bb" else None,
                "atbb_age_guard_ms": float(spec.get("atbb_age_guard_ms", 1_000.0)) if spec["policy"] == "at-bb" else None,
                "atbb_sample_count": int(spec.get("atbb_sample_count", 64)) if spec["policy"] == "at-bb" else None}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    engine = LLMSingleNodeSimulator(requests, config, scheduler)
    summary = engine.run(max_steps=int(spec.get("max_steps", 200_000)),
                         wall_time_limit_s=float(spec.get("wall_time_limit_s", 120)))
    rows = request_rows(engine, scheduler)
    complete = all(r["complete"] for r in rows)
    reference_path = (root / spec["reference"]).resolve() if spec.get("reference") else None
    if reference_path and (identity["model"] == "unprofiled" or identity["hardware"] == "unprofiled"):
        raise ValueError("GPU comparison requires explicit model and hardware identity")
    if reference_path and config.execution_mode != "scheduled-tokens":
        raise ValueError("boundary mode cannot validate token-level measurements")
    comparison = compare_reference(reference_path, signature, rows, complete)
    service_window = engine.now_ms - min(r.arrival_ms for r in requests)
    metrics = {"completed": sum(r["complete"] for r in rows), "requests": len(rows),
               "termination": summary["termination_reason"], "iterations": engine.simulation_steps,
               "service_window_ms": service_window,
               "realized_throughput_rps": sum(r["complete"] for r in rows) * 1000 / service_window if service_window > 0 else None,
               "peak_kv_util": engine.peak_kv_tokens / config.kv_capacity_tokens,
               "evictions": sum(r["evictions"] for r in rows),
               "admission_rejections": scheduler.admission_rejections,
               "s3_preemptions": scheduler.s3_preemptions,
               "reservation_episodes": int(getattr(scheduler, "reservation_episodes", 0)),
               "realized_backfills": int(getattr(scheduler, "realized_backfills", 0))}
    metrics["timing_out_of_domain_iterations"] = sum(bool(s.get("timing_domain_warnings")) for s in scheduler.steps)
    metrics["timing_out_of_domain_fraction"] = metrics["timing_out_of_domain_iterations"] / max(1, len(scheduler.steps))
    warning_counts = Counter(
        warning
        for step in scheduler.steps
        for warning in str(step.get("timing_domain_warnings", "")).split(",")
        if warning
    )
    metrics["timing_domain_warning_counts"] = dict(sorted(warning_counts.items()))
    for metric in ("ttft_ms", "e2e_ms", "per_token_latency_ms", "tbt_max_ms"):
        values = [float(row[metric]) for row in rows if row[metric] is not None]
        metrics[f"{metric}_mean"] = statistics.fmean(values) if values else None
        metrics[f"{metric}_p50"] = percentile(values, 50) if values else None
        metrics[f"{metric}_p95"] = percentile(values, 95) if values else None
        metrics[f"{metric}_p99"] = percentile(values, 99) if values else None
    max_waiting = [max(float(row["ttft_ms"]), float(row["tbt_max_ms"] or 0.0))
                   for row in rows if row["ttft_ms"] is not None]
    metrics["max_waiting_time_ms_mean"] = statistics.fmean(max_waiting) if max_waiting else None
    metrics["max_waiting_time_ms_p95"] = percentile(max_waiting, 95) if max_waiting else None
    metrics["max_waiting_time_ms_max"] = max(max_waiting) if max_waiting else None
    ttfts = [float(row["ttft_ms"]) for row in rows if row["ttft_ms"] is not None]
    scheduling_delays = [float(row["scheduling_delay_ms"]) for row in rows
                         if row["scheduling_delay_ms"] is not None]
    paper_tbts = paper_token_tbt_values(scheduler) if config.execution_mode == "scheduled-tokens" else []
    metrics["paper_ttft_p50_ms"] = percentile(ttfts, 50) if ttfts else None
    metrics["paper_scheduling_delay_p50_ms"] = percentile(scheduling_delays, 50) if scheduling_delays else None
    metrics["paper_tbt_all_decode_tokens_p99_ms"] = percentile(paper_tbts, 99) if paper_tbts else None
    arrival_window = max(r.arrival_ms for r in requests) - min(r.arrival_ms for r in requests)
    metrics["offered_load_rps"] = ((len(requests) - 1) * 1000 / arrival_window
                                   if len(requests) > 1 and arrival_window > 0 else None)
    metrics["paper_capacity_slo_met"] = (None if criteria is None else bool(
        complete
        and metrics["paper_scheduling_delay_p50_ms"] <= criteria["p50_scheduling_delay_slo_ms"]
        and metrics["paper_tbt_all_decode_tokens_p99_ms"] <= criteria["p99_tbt_slo_ms"]
    ))
    output.mkdir(parents=True, exist_ok=True)
    if bool(spec.get("write_details", True)):
        write_csv(output / "requests.csv", rows)
        write_csv(output / "iterations.csv", scheduler.steps)
        write_csv(output / "measurement_errors.csv", comparison.get("request_deltas", []))
        template = {"source_kind": "UNSET", "source": "", "measurement_role": "validation",
                    "experiment_signature": signature, "calibration_request_ids": [],
                    "requests": [{"request_id": r["request_id"], "ttft_ms": None, "e2e_ms": None,
                                  "tbt_max_ms": None} for r in rows]}
        (output / "reference_template.json").write_text(json.dumps(template, indent=2), encoding="utf-8")
    result = {"name": spec["name"], "experiment_signature": signature, "identity": identity,
              "implementation_sha256": {name: digest(Path(__file__).resolve().parent / name)
                                        for name in IMPLEMENTATION_FILES},
              "metrics": metrics, "validation": comparison}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def render_report(results: list[dict]) -> str:
    lines = ["# 正式会议论文调度算法本机对照实验", "",
             "本报告在同一 trace、KV allocator 和解析时间模型中运行论文调度规则。它是算法级 matched replay，不是论文 serving engine 的 GPU kernel 复刻；缺少同配置 GPU 日志时不报告模拟准确率。", "",
             "| 实验 | 策略来源 | 完成 | TTFT P50/P99 ms | E2E mean/P99 ms | TBT P99 ms | max-wait P95 ms | 吞吐 req/s | KV 峰值 | 驱逐 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for result in results:
        m = result["metrics"]
        def number(key):
            value = m[key]
            return f"{value:.3f}" if value is not None else "N/A"
        lines.append(f"| {result['name']} | {result['identity']['publication_provenance']} | {m['completed']}/{m['requests']} | "
                     f"{number('ttft_ms_p50')} / {number('ttft_ms_p99')} | "
                     f"{number('e2e_ms_mean')} / {number('e2e_ms_p99')} | {number('tbt_max_ms_p99')} | "
                     f"{number('max_waiting_time_ms_p95')} | {number('realized_throughput_rps')} | "
                     f"{m['peak_kv_util']:.1%} | {m['evictions']} |")
    by_name = {result["name"]: result for result in results}
    relative = [(result, by_name.get(result["identity"].get("baseline_name"))) for result in results
                if result["identity"].get("baseline_name")]
    if relative:
        lines += ["", "## 相对 FCFS", "",
                  "正值表示相对同 workload 的 FCFS 改善；延迟按 `(FCFS-policy)/FCFS`，吞吐按 `(policy/FCFS)-1`。Oracle 行是使用隐藏真值的不可部署信息对照，不保证含抢占成本的端到端指标构成数学上界。", "",
                  "| 实验 | TTFT P99 | E2E mean | E2E P99 | max-wait P95 | 吞吐 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for result, baseline in relative:
            if baseline is None:
                continue
            current, base = result["metrics"], baseline["metrics"]
            def latency_gain(key):
                return 100 * (base[key] - current[key]) / base[key] if base[key] else None
            def throughput_gain():
                return 100 * (current["realized_throughput_rps"] / base["realized_throughput_rps"] - 1)
            def percent(value):
                return f"{value:+.2f}%" if value is not None else "N/A"
            oracle = " (Oracle)" if result["identity"]["uses_hidden_output_for_scheduling"] else ""
            lines.append(f"| {result['name']}{oracle} | {percent(latency_gain('ttft_ms_p99'))} | "
                         f"{percent(latency_gain('e2e_ms_mean'))} | {percent(latency_gain('e2e_ms_p99'))} | "
                         f"{percent(latency_gain('max_waiting_time_ms_p95'))} | {percent(throughput_gain())} |")
    groups = sorted({result["identity"].get("comparison_group") for result in results
                     if result["identity"].get("comparison_group")})
    if groups:
        lines += ["", "## 结果解读", "",
                  "以下只在不读取隐藏输出真值的策略中选择最佳值。不同列可能由不同策略取得，因此不能合并成单一的全面胜者。", "",
                  "| workload | 最低 E2E mean | 相对 FCFS | 最低 TTFT P99 | 相对 FCFS |",
                  "| --- | --- | ---: | --- | ---: |"]
        best_rows = []
        for group in groups:
            members = [result for result in results
                       if result["identity"].get("comparison_group") == group
                       and not result["identity"]["uses_hidden_output_for_scheduling"]]
            baseline = next((result for result in members if result["name"] == f"{group}_fcfs"), None)
            if baseline is None:
                continue
            mean_best = min(members, key=lambda result: result["metrics"]["e2e_ms_mean"])
            tail_best = min(members, key=lambda result: result["metrics"]["ttft_ms_p99"])
            mean_gain = 100 * (baseline["metrics"]["e2e_ms_mean"] - mean_best["metrics"]["e2e_ms_mean"]) / baseline["metrics"]["e2e_ms_mean"]
            tail_gain = 100 * (baseline["metrics"]["ttft_ms_p99"] - tail_best["metrics"]["ttft_ms_p99"]) / baseline["metrics"]["ttft_ms_p99"]
            lines.append(f"| {group} | {mean_best['name']} ({mean_best['metrics']['e2e_ms_mean']:.3f} ms) | "
                         f"{mean_gain:+.2f}% | {tail_best['name']} ({tail_best['metrics']['ttft_ms_p99']:.3f} ms) | "
                         f"{tail_gain:+.2f}% |")
            best_rows.append((group, mean_best, tail_best))
        lines += ["",
                  "LTR 的低中位 TTFT 来自短请求优先，但当前 KV 压力下会造成大量驱逐和长暂停；因此必须同时看 P99 TTFT、P99 TBT、E2E 和驱逐数，不能只报中位数。"]
        for group, mean_best, tail_best in best_rows:
            if mean_best is tail_best:
                metrics = mean_best["metrics"]
                details = [f"{metrics['evictions']} 次驱逐"]
                if metrics.get("realized_backfills"):
                    details.append(f"{metrics['realized_backfills']} 次实际回填")
                lines.append(
                    f"{group} 中，{mean_best['name']} 同时取得最低的可部署 E2E mean 和 TTFT P99"
                    f"（{'、'.join(details)}）。"
                )
            else:
                lines.append(
                    f"{group} 中，最低可部署 E2E mean 来自 {mean_best['name']}，"
                    f"最低可部署 TTFT P99 来自 {tail_best['name']}。"
                )
    comparisons = [(name, result, by_name[name + "_boundary"]) for name, result in by_name.items()
                   if name + "_boundary" in by_name]
    if comparisons:
        lines += ["", "## 执行语义差异", "",
                  "以下是同一策略在本模拟器两种执行模式下的差异，反映阶段压缩的影响；它不是相对真实 GPU 的误差。", "",
                  "| 实验 | 原模式 E2E ms | 逐 token 模式 E2E ms | 变化 | 原模式驱逐数 | 逐 token 驱逐数 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, current, old in comparisons:
            a, b = old["metrics"]["e2e_ms_mean"], current["metrics"]["e2e_ms_mean"]
            if a is not None and b is not None and a > 0:
                lines.append(f"| {name} | {a:.3f} | {b:.3f} | {(b/a-1)*100:+.3f}% | "
                             f"{old['metrics']['evictions']} | {current['metrics']['evictions']} |")
    lines += ["", "## 论文与状态", "",
              "- Orca: OSDI 2022, https://www.usenix.org/conference/osdi22/presentation/yu",
              "- vLLM: SOSP 2023, https://doi.org/10.1145/3600006.3613165",
              "- S3: NeurIPS 2023, https://proceedings.neurips.cc/paper_files/paper/2023/hash/3a13be0c5dae69e0f08065f113fb10b8-Abstract-Conference.html",
              "- Learning to Rank (LTR): NeurIPS 2024, https://proceedings.neurips.cc/paper_files/paper/2024/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html",
              "- Sarathi-Serve: OSDI 2024, https://www.usenix.org/conference/osdi24/presentation/agrawal",
              "- AT-BB: 本仓库 Active-Tail Budgeted Backfilling 方法。",
              "- MLFQ 是传统通用算法，本实验按 LTR 论文附录中的 baseline 参数实现，不将其描述为一篇独立录用论文的贡献。",
              "- Past-Future 已核实发表于 ASPLOS 2025，DOI: 10.1145/3676641.3716011；旧预印本标注已纠正。TIE 仍单列，本文不在没有来源核实的情况下判断其他论文的发表状态。", "",
              "## 复现边界", "",
              "- Orca/vLLM：`fcfs` 复现 iteration-level FCFS；`orca-max` 为最大输出预留，`orca-pow2` 和 `orca-oracle` 读取真实长度，只用于 vLLM 论文中的离线内存上界对照。",
              "- S3：复现 10 桶长度预测、按预测 KV 降序 first-fit、超过预留后抢占并将预留翻倍；预测值来自 trace 内调度器可见的请求长度分布，抢占复用本仓库的丢弃/重算路径，未模拟论文的异步 host swap。",
              "- LTR：复现 Algorithm 1 的每轮 `(priority, score)` 排序、抢占式 batch 选择和 starvation quantum。`ltr` 使用共享可见长度预测，所以是 scheduling-rule reproduction，不声称复现论文 OPT checkpoint；`ltr-oracle` 只作真值信息诊断。",
              "- MLFQ：按 LTR 附录采用 16 秒 base quantum、2 倍增长；batch 上限统一使用本实验配置，而不是论文的 256。",
              "- Sarathi：接入 Algorithm 3 的 decode 优先、未完成 prefill 优先、剩余 token budget 分块；基线 fcfs 对应论文 Algorithm 2 的 prefill 优先。",
              "- AT-BB：运行中请求使用 0.75 固定上尾释放预测；队首等待 1000 ms 且被预测 KV gate 阻塞时，允许一个预测服务工作不超过 100 ms 的安全候选回填，并预约原队首。",
              "- 本套件为缺少显式请求上限的 trace 统一设置 `max_output_tokens=2048`，回放输出也按同一上限截断，避免从真实输出反推调度可见上限。",
              "- scheduled-tokens 模式执行调度器指定的 token 数；boundary 为仓库原有的推进到阶段结束/KV 边界模式，用于执行语义对照。",
              "- 论文同口径指标包括 TTFT P50、请求首次被调度前的 scheduling delay P50，以及所有 decode token 的 TBT P99；按作者实现，每个请求的第一个输出 token 贡献一个 0 值 TBT。capacity_criteria 可同时检查 2 秒 scheduling-delay 阈值和指定 TBT SLO。",
              "- 当前时间成本由各实验 timing 配置决定；首 token 约定见 result.json 的 first_token_convention（prefill 或 decode）。接入规则不等于完成了原 serving engine 的复现。",
              "- 本地 trace 不等于论文原始负载；这些运行验证接入、机制和执行语义，不提供实机准确率。",
              "- measurement_errors.csv 保存逐请求误差；result.json 保存签名、MAE、MAPE、RMSE、R²。签名不匹配、缺请求或复用校准请求会拒绝比较。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/paper_simulator_validation.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper_simulator_validation_20260907"))
    args = parser.parse_args()
    payload = json.loads(args.suite.read_text(encoding="utf-8"))
    root = args.suite.resolve().parent / payload.get("path_root", "..")
    defaults = payload.get("defaults", {})
    specs = [{**defaults, **spec,
              "simulator": {**defaults.get("simulator", {}), **spec.get("simulator", {})}}
             for spec in payload["experiments"]]
    names = [s["name"] for s in specs]
    if not specs or len(names) != len(set(names)) or any(Path(n).name != n or n in {".", ".."} for n in names):
        raise ValueError("experiment names must be unique directory names")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in specs:
        result = run_case(spec, root, args.output_dir / spec["name"])
        results.append(result)
        print(json.dumps({"name": spec["name"], **result["metrics"]}), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()
