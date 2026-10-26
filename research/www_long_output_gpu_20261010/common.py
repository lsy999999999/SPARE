"""Constants for a Conversation long-output GPU stress test."""
import os
from pathlib import Path

os.environ.setdefault("WWW_GPU_OUT", "artifacts/www_long_output_gpu_20261010_r1")

from research.www_burst_background_gpu_20261009 import common as base

ROOT = base.ROOT
HERE = Path(__file__).resolve().parent
OUT = base.OUT
MODEL = base.MODEL
PYTHON = base.PYTHON
read = base.read
write = base.write
digest = base.digest
object_digest = base.object_digest
ARMS = base.ARMS
OPTIONS = base.OPTIONS
METHOD = base.METHOD
ORACLE_ARMS = base.ORACLE_ARMS


def source_paths():
    paths = base.source_paths()
    paths.update({str(path.relative_to(ROOT)): path for path in HERE.glob("*.py")})
    protocol = HERE / "PROTOCOL_CN.md"
    paths[str(protocol.relative_to(ROOT))] = protocol
    return paths


def source_hashes():
    return {logical: digest(path) for logical, path in sorted(source_paths().items())}
