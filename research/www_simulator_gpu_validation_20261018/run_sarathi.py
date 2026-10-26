"""Run the frozen Sarathi supplement with the original GPU measurement logic."""
from __future__ import annotations

import argparse
import time

from research.www_simulator_gpu_validation_20261018 import run as base


base.OUT = base.ROOT / "artifacts/www_simulator_gpu_validation_sarathi_20261018_r1"
base.gpu_run.OUT = base.OUT
base.pilot.OUT = base.OUT


def preflight(gpu):
    frozen = base.design()
    handle = base.lock_gpu(gpu)
    try:
        import torch

        torch.cuda.set_device(0)
        engine, scheduler = base.pilot.engine_for(2048)
        requests = [dict(id=f"probe{i}", prompt=256 + i * 64,
                         output=4, arrival_ms=float(i * 30)) for i in range(8)]
        cell = dict(budget=frozen["budget"], chunk=frozen["chunk"],
                    history_file="history/code.json")
        result = base.gpu_run.run_episode(engine, scheduler, requests,
                                          "sarathi", cell, 120)
        if result["status"] != "complete":
            raise RuntimeError(result["error"])
        base.write(base.OUT / "preflight.json", dict(
            passed=True, gpu=gpu, design_sha256=base.sha(base.OUT / "design.json"),
            finished=time.time()))
    finally:
        handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight(args.gpu)
    else:
        base.run(args.gpu)
