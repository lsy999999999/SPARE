"""Compact remote status for the idea1 real-GPU pilot."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_idea1_gpu_pilot_20261016_r1"


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


if __name__ == "__main__":
    design = read(OUT / "design.json")
    launch = read(OUT / "launch.json")
    status = read(OUT / "status.json")
    pid = launch.get("pid")
    runs = list((OUT / "rates").glob("*.json"))
    print("Idea1 real RTX 4090 GPU pilot")
    print(f"state={status.get('state', 'not_started')} pid={pid} "
          f"alive={bool(pid and Path(f'/proc/{pid}').exists())}")
    print(f"groups={status.get('completed_groups', 0)}/{status.get('total_groups', 0)} "
          f"rate_runs={len(runs)} failures={len(status.get('failures', []))}")
    print("active=" + json.dumps(status.get("active", {}), sort_keys=True))
    print(f"methods={len(design.get('methods', []))} "
          f"KV={design.get('capacities', [])} updated={status.get('updated', '-')}")
    for path in sorted((OUT / "current").glob("gpu*.json")):
        print(path.stem, json.dumps(read(path), sort_keys=True))
    if status.get("failures"):
        print("recent_failure=" + json.dumps(status["failures"][-1], sort_keys=True))
