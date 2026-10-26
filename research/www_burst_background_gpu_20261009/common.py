"""Frozen constants for the burst followed by background GPU replay."""
import os
from pathlib import Path

os.environ.setdefault("WWW_GPU_OUT", "artifacts/www_burst_background_gpu_20261009_r2")

from research.www_evaluation_gpu_completion_20261003 import common as base

ROOT = base.ROOT
HERE = Path(__file__).resolve().parent
OUT = base.OUT
MODEL = base.MODEL
PYTHON = base.PYTHON
read = base.read
write = base.write
digest = base.digest
object_digest = base.object_digest

ARMS = ("arrival", "srp", "fixed256", "fixed512")
OPTIONS = {"arrival": {}, "fixed256": {"mode": "fixed", "fixed_budget": 256},
           "fixed512": {"mode": "fixed", "fixed_budget": 512}}
METHOD = {"arrival": "threshold", "srp": "srp", "native": "native",
          "fixed256": "threshold", "fixed512": "threshold"}
ORACLE_ARMS = ()


def source_paths():
    paths = base.source_paths()
    paths.update({str(path.relative_to(ROOT)): path for path in HERE.glob("*.py")})
    protocol = HERE / "PROTOCOL_CN.md"
    paths[str(protocol.relative_to(ROOT))] = protocol
    return paths


def source_hashes():
    return {logical: digest(path) for logical, path in sorted(source_paths().items())}
