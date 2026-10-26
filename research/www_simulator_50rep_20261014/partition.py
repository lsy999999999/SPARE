"""Run a disjoint subset of frozen trace samples on one CPU host."""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from research.www_simulator_50rep_20261014 import run as batch
from research.www_evaluation_completion_20261003.common import ROOT, write, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-start", type=int, required=True)
    parser.add_argument("--sample-stop", type=int, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--host", choices=("local", "remote"), required=True)
    args = parser.parse_args()
    if not (0 <= args.sample_start < args.sample_stop <= batch.SAMPLE_COUNT):
        parser.error("invalid half-open sample range")
    design = batch.prepare()
    for path, expected in design["hashes"].items():
        if args.host == "remote" and path.startswith("data/azure_"):
            continue  # Frozen samples and their hashes were transferred from the local host.
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Frozen source/input changed: {path}")
    groups = defaultdict(list)
    for task in design["tasks"]:
        groups[task["window"]].append(task)
    samples = {(item["window"], item["sample"]): item["path"]
               for item in design["samples"]}
    work = [(sample, repeat, window, groups[window], samples[window, sample])
            for sample in range(args.sample_start, args.sample_stop)
            for repeat in range(batch.REPEATS_PER_SAMPLE)
            for window in design["windows"]
            if not batch.accepted(sample, repeat, window, len(groups[window]))]
    expected = (args.sample_stop - args.sample_start) * batch.REPEATS_PER_SAMPLE * len(design["windows"])
    status_path = batch.OUT / f"partition_status_{args.host}.json"
    start = time.monotonic()
    initial = expected - len(work)
    finished = initial
    failures = []

    def publish(stage: str, remaining: int) -> None:
        write(status_path, dict(stage=stage, host=args.host,
                                sample_start=args.sample_start, sample_stop=args.sample_stop,
                                jobs=args.jobs, accepted_shards=finished,
                                expected_shards=expected, remaining=remaining,
                                failures=failures, elapsed_s=time.monotonic() - start))

    publish("running", len(work))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(batch.run_shard, task): task[:3] for task in work}
        while futures:
            ready, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for future in ready:
                sample, repeat, window = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(sample=sample, repeat=repeat, window=window,
                                  complete=False, error=repr(exc))
                if result["complete"] and batch.accepted(sample, repeat, window, len(groups[window])):
                    finished += 1
                else:
                    failures.append(result)
                    print("FAILED", result, flush=True)
            publish("running", len(futures))
    publish("complete" if not failures and finished == expected else "incomplete", 0)
    print(args.host, finished, "/", expected, "shards; failures", len(failures), flush=True)


if __name__ == "__main__":
    main()
