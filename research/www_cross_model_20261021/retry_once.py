"""Retry missing groups once after the current GPU controller exits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from research.www_cross_model_20261021 import campaign


def controller_alive(pid: int) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return b"www_cross_model_20261021/campaign.py" in command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    while controller_alive(args.wait_pid):
        time.sleep(30)

    out = campaign.OUT
    status = json.loads((out / "status.json").read_text())
    if status["state"] != "finished_with_failures":
        print(f"No retry: controller state is {status['state']}", flush=True)
        return
    missing = [f"{model}_{source}_c{kv}"
               for model, kvs in campaign.MODELS.items()
               for kv in kvs for source in ("code", "conversation")
               if not (out / "groups" / f"{model}_{source}_c{kv}.json").exists()]
    if not missing:
        print("No missing groups", flush=True)
        return
    log = out / "controller_retry.log"
    env = dict(os.environ, PYTHONPATH=str(campaign.ROOT),
               WWW_CROSS_MODEL_ROOT=str(campaign.ROOT))
    with log.open("a") as stream:
        process = subprocess.Popen(
            [campaign.PYTHON, "-B", "-u", str(Path(campaign.__file__))],
            cwd=campaign.ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    campaign.write(out / "retry_launch.json",
                   {"pid": process.pid, "started": time.time(), "missing": missing})
    print(f"Retry controller {process.pid}: {len(missing)} groups", flush=True)


if __name__ == "__main__":
    main()
