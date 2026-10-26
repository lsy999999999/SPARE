"""Validate Code long-output inputs and actual scheduler paths."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_code_long_output_gpu_20261010 import common as spec
from research.www_long_output_gpu_20261010 import validate as base

base.OUT = spec.OUT
base.source_hashes = spec.source_hashes
base.main()
