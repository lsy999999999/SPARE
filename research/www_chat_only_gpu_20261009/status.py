"""Print the Conversation-only controller status."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_chat_only_gpu_20261009 import common as spec
from research.www_evaluation_gpu_completion_20261003 import status as base

base.OUT = spec.OUT
base.main()
