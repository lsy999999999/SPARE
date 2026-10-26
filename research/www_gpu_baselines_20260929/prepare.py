"""Freeze the unique GPU scenarios represented by the two simulator profiles."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_gpu_baselines_20260929.common import *


def main():
    if (OUT / "design.json").exists():
        raise SystemExit("design already frozen; use a new output directory")
    suite = read(ROOT / "writing/www/experiments/suite.json")
    if tuple(suite["arms"]) != ARMS:
        raise ValueError("simulator arm list changed")
    grouped = {}
    for case in suite["cases"]:
        key = (case["input"], case["capacity"], case.get("budget", 1024),
               case.get("chunk", 256), case["history"])
        grouped.setdefault(key, []).append(case)
    if len(grouped) != 16:
        raise ValueError(f"expected 16 unique GPU scenarios, found {len(grouped)}")
    cells = []
    for index, (key, simulator_cases) in enumerate(sorted(grouped.items())):
        source, capacity, budget, chunk, history = key
        profiles = sorted(case["profile"] for case in simulator_cases)
        if profiles != ["resource", "resource_staircase"]:
            raise ValueError((source, profiles))
        requests = read(ROOT / source)
        if len(requests) != 64:
            raise ValueError((source, len(requests)))
        input_name = f"cell{index:02d}_{source.rsplit('/', 1)[-1]}"
        input_target = OUT / "inputs" / input_name
        input_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, input_target)
        history_rows = [json.loads(line) for line in (ROOT / history).read_text().splitlines()
                        if line.strip()]
        lengths = [int(row["actual_output_tokens"]) for row in history_rows]
        history_target = OUT / "history" / f"cell{index:02d}.json"
        write(history_target, dict(source=history, source_sha256=digest(ROOT / history),
                                   lengths=lengths))
        cells.append(dict(
            key=f"cell{index:02d}_{Path(source).stem}_c{capacity}",
            source_input=source, input=str(input_target.relative_to(OUT)),
            input_sha256=digest(input_target), history=history,
            history_file=str(history_target.relative_to(OUT)),
            history_sha256=digest(ROOT / history), capacity=capacity,
            budget=budget, chunk=chunk, requests=64,
            simulator_cases=[case["name"] for case in simulator_cases],
        ))
    design = dict(
        schema=1, frozen=True, frozen_at=time.time(), cells=cells,
        methods=list(ARMS), repeats=4, aa_repeats=2,
        expected_formal_runs=len(cells) * len(ARMS) * 4,
        expected_aa_runs=len(cells) * 2, max_model_len=8176,
        max_num_seqs=64, max_num_batched_tokens=2048, block_size=16,
        dtype="float16", gpu_memory_utilization=.9, enforce_eager=True,
        enable_prefix_caching=False, enable_chunked_prefill=True,
        episode_timeout_s=600, deadline_hours=24,
        gpu_candidates=[0, 1, 2, 3, 6, 7], max_group_retries=2,
        ordering="two deterministic shuffled/reverse pairs per cell",
        objective="mean TTFT primary; P95/P99 TTFT, E2E, ITL, throughput, SLO, KV and recovery secondary",
        stopping="fixed repetitions and 24-hour hard deadline; no outcome-dependent stopping",
        source_hashes=source_hashes(),
        interpretation=("GPU execution of shared-vLLM policy/rule adaptations. "
                        "Oracle/proxy arms use realized output lengths and are not deployable. "
                        "This is not a reproduction of each paper's original serving stack."),
    )
    write(OUT / "design.json", design)
    for relative in design["source_hashes"]:
        target = OUT / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    print(json.dumps(dict(cells=len(cells), formal_runs=design["expected_formal_runs"],
                          aa_runs=design["expected_aa_runs"], output=str(OUT)), indent=2))


if __name__ == "__main__":
    main()
