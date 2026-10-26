"""Export the concise architecture figure to editable PowerPoint objects."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent
source = ROOT.parent / 'architecture_v2' / 'export_powerpoint.py'
spec = importlib.util.spec_from_file_location('spare_powerpoint', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT = ROOT
module.main()
