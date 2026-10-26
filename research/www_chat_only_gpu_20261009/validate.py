"""Validate inputs and the five GPU execution paths before launch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_chat_only_gpu_20261009 import common as spec
from research.www_burst_background_gpu_20261009 import validate as base

base.OUT = spec.OUT
base.source_hashes = spec.source_hashes
base.main()
