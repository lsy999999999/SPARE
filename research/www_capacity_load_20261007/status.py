"""Human-readable progress for the detached WWW GPU baseline controller."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_capacity_load_20261007.common import *


def timestamp(value):
    if value is None:
        return "-"
    return dt.datetime.fromtimestamp(value).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def main():
    design = read(OUT / "design.json")
    launch = read(OUT / "launch.json") if (OUT / "launch.json").exists() else {}
    status = read(OUT / "status.json") if (OUT / "status.json").exists() else {}
    pid = launch.get("pid")
    alive = bool(pid and Path(f"/proc/{pid}").exists())
    attempts = list((OUT / "attempts").glob("*/a*/runs/*.json"))
    formal = 0
    aa = 0
    failed_runs = 0
    for path in attempts:
        row = read(path)
        if row.get("phase") == "formal" and row.get("status") == "complete":
            formal += 1
        elif row.get("phase") == "aa" and row.get("status") == "complete":
            aa += 1
        if row.get("status") != "complete":
            failed_runs += 1
    groups = list((OUT / "groups").glob("*.json"))
    complete_groups = sum(read(path).get("state") == "complete" for path in groups)
    now = time.time()
    print("WWW GPU baseline matrix")
    print(f"state={status.get('state', 'not_started')} controller_pid={pid} alive={alive}")
    print(f"started={timestamp(launch.get('started'))}")
    print(f"deadline={timestamp(launch.get('deadline'))} remaining_hours={max(0., (launch.get('deadline', now)-now)/3600):.2f}")
    print(f"groups={complete_groups}/{len(design['cells'])} formal_runs={formal}/{design['expected_formal_runs']} aa_runs={aa}/{design['expected_aa_runs']} failed_run_files={failed_runs}")
    active = status.get("active", {})
    if active:
        print("active:")
        for gpu, value in sorted(active.items(), key=lambda item: int(item[0])):
            current_path = OUT / value["folder"] / "current.json"
            current = read(current_path) if current_path.exists() else {}
            print(f"  gpu{gpu}: {value['cell']} attempt={value['attempt']} current={current.get('name', 'initializing')} state={current.get('state', '-')}")
    waiting = []
    for path in sorted(OUT.glob("worker_g*.json")):
        row = read(path)
        if row.get("state") == "waiting_for_idle_gpu":
            waiting.append(f"{path.stem.removeprefix('worker_')} pids={row.get('compute_pids', [])}")
    if waiting:
        print("occupied candidates: " + "; ".join(waiting))
    failures = [read(path) for path in (OUT / "groups").glob("*.json")
                if read(path).get("state") != "complete"]
    if failures:
        print(f"terminal_failed_groups={len(failures)}")
    print(f"updated={timestamp(status.get('updated'))}")


if __name__ == "__main__":
    main()
