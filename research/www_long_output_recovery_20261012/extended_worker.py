"""Run the frozen worker with only the formal episode deadline amended."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_long_output_gpu_20261010.run import core

EXTENDED_EPISODE_TIMEOUT_S = 21600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    if not args.folder.endswith("/a5"):
        raise RuntimeError("extended deadline is only authorized for a5")
    original_timeout = core.verify_design()["episode_timeout_s"]
    original_save_run = core.save_run

    def save_run(engine, scheduler, requests, arm, cell, folder, name, metadata, timeout):
        if timeout != original_timeout:
            raise RuntimeError(f"unexpected formal episode timeout: {timeout}")
        return original_save_run(engine, scheduler, requests, arm, cell, folder, name,
            dict(metadata, effective_episode_timeout_s=EXTENDED_EPISODE_TIMEOUT_S),
            EXTENDED_EPISODE_TIMEOUT_S)

    core.save_run = save_run
    core.worker(args.worker, args.gpu, args.folder)


if __name__ == "__main__":
    main()
