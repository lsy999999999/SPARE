"""Frozen matrix constants for the fixed-256 explanation experiment."""
import os
from pathlib import Path

os.environ.setdefault("WWW_GPU_OUT", "artifacts/www_fixed256_explanation_20261014_r1")
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

ARMS = ("full", "fixed256", "accept256", "recovery256", "no_finish")
OPTIONS = {
    "full": {},
    "fixed256": {"mode": "fixed", "fixed_budget": 256},
    "accept256": {"mode": "accept256"},
    "recovery256": {"mode": "recovery256"},
    "no_finish": {"mode": "no_finish", "protect_finish": False},
}
METHOD = {arm: "threshold" for arm in ARMS}
METHOD["native"] = "native"
ORACLE_ARMS = ()


def source_paths():
    paths = base.source_paths()
    paths.update({str(path.relative_to(ROOT)): path for path in HERE.glob("*.py")})
    return paths


def source_hashes():
    return {logical: digest(path) for logical, path in sorted(source_paths().items())}
