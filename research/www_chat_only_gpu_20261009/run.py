"""Launch the frozen Conversation-only matrix on the audited GPU controller."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_chat_only_gpu_20261009 import common as spec
from research.www_evaluation_gpu_completion_20261003 import run as core

for name in ("OUT", "HERE", "ARMS", "OPTIONS", "METHOD", "ORACLE_ARMS",
             "source_hashes", "source_paths"):
    setattr(core, name, getattr(spec, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--folder")
    args = parser.parse_args()
    if args.launch:
        core.launch()
    elif args.worker:
        core.worker(args.worker, args.gpu, args.folder)
    else:
        try:
            core.controller()
        except BaseException:
            spec.write(spec.OUT / "status.json", dict(state="failed",
                error=core.traceback.format_exc(), updated=core.time.time()))
            raise


if __name__ == "__main__":
    main()
