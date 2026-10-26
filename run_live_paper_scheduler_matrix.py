#!/usr/bin/env python3
"""Run the deployable paper-policy gateway adaptations against a live vLLM server."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VALIDATOR = (
    ROOT
    / "research/kv_risk_feasibility_20260909/recovery_cost_v2/gpu_serving_validation.py"
)
MODEL = "models/Qwen2.5-7B-Instruct"
WORKLOADS = {
    "chat": (
        "workloads/castillo_azure_conditional_v1/test.jsonl",
        "workloads/castillo_azure_conditional_v1/train.jsonl",
        1.0,
    ),
    "code": (
        "workloads/castillo_azure_code_conditional_v1/test.jsonl",
        "workloads/castillo_azure_code_conditional_v1/train.jsonl",
        1.0,
    ),
    "burst0": (
        "research/kv_risk_feasibility_20260909/recovery_cost_v2/"
        "survival_tail_final_results/workloads/segment_0_160.jsonl",
        "research/kv_risk_feasibility_20260909/recovery_cost_v2/"
        "burstgpt_chronological_workload/train.jsonl",
        60_000.0,
    ),
    "burst3": (
        "research/kv_risk_feasibility_20260909/recovery_cost_v2/"
        "survival_tail_final_results/workloads/segment_3_160.jsonl",
        "research/kv_risk_feasibility_20260909/recovery_cost_v2/"
        "burstgpt_chronological_workload/train.jsonl",
        60_000.0,
    ),
}
NORMAL_POLICIES = (
    ("fcfs", "FCFS"),
    ("orca_max", "Orca-Max"),
    ("s3", "S3"),
    ("ltr", "LTR"),
    ("mlfq", "MLFQ"),
    ("at_bb", "Static-Tail-75-Budgeted"),
)


def run_one(args: argparse.Namespace, repetition: int, label: str, policy: str) -> None:
    output = args.output_dir / f"{args.workload}_{label}_rep{repetition}.json"
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("result", {}).get("valid"):
            print(f"reuse {output.name}", flush=True)
            return
    trace, train, arrival_scale = WORKLOADS[args.workload]
    command = [
        sys.executable,
        str(VALIDATOR),
        "--base-url", args.base_url,
        "--model", MODEL,
        "--policy", policy,
        "--trace", trace,
        "--train-trace", train,
        "--requests", str(args.requests),
        "--arrival-scale", str(arrival_scale),
        "--max-prompt-tokens", "512",
        "--max-output-tokens", "256",
        "--max-inflight", "8",
        "--external-kv-capacity-tokens", "4000",
        "--prefill-ms-per-token", "0.2476",
        "--decode-ms-per-token", "2.7646",
        "--delay-budget-ms", "500",
        "--distribution-model", "empirical-knn",
        "--knn-neighbors", "64",
        "--output", str(output),
    ]
    print(
        f"run workload={args.workload} policy={label} repetition={repetition}",
        flush=True,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode:
        print(completed.stdout[-8000:], flush=True)
        completed.check_returncode()
    result = json.loads(output.read_text(encoding="utf-8"))["result"]
    if not result["valid"] or result["completed"] != args.requests:
        raise RuntimeError(f"invalid result: {output}")
    print(
        f"done {label} rep={repetition}: "
        f"E2E={result['replay_e2e_mean_ms']:.1f} ms, "
        f"TTFT-P99={result['replay_ttft_p99_ms']:.1f} ms, "
        f"throughput={result['request_throughput_rps']:.3f} req/s, "
        f"preemptions={result['vllm_preemptions']:.0f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--workload", choices=tuple(WORKLOADS), required=True)
    parser.add_argument("--phase", choices=("normal", "sarathi"), required=True)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policies = NORMAL_POLICIES if args.phase == "normal" else (("sarathi", "Sarathi"),)
    for repetition in range(args.repetitions):
        ordered = policies if repetition % 2 == 0 else tuple(reversed(policies))
        for label, policy in ordered:
            run_one(args, repetition, label, policy)


if __name__ == "__main__":
    main()
