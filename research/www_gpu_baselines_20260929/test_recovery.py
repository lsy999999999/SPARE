"""Regression checks for failures found in the first full GPU batch."""
import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from research.www_gpu_baselines_20260929 import run


class ProcRaceTest(unittest.TestCase):
    def test_disappearing_proc_entry_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            child = root / "20"
            child.mkdir()
            (child / "stat").write_text("20 (worker name) S 10 0 0\n")
            vanished = root / "21"
            vanished.mkdir()
            vanished.rmdir()
            self.assertEqual(run.descendants(10, root), {10, 20})

    def test_scandir_race_returns_best_effort_snapshot(self):
        class RacingScan:
            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def __iter__(self):
                return self

            def __next__(self):
                raise ProcessLookupError("process vanished")

        with mock.patch.object(run.os, "scandir", return_value=RacingScan()):
            self.assertEqual(run.descendants(10), {10})


class PriorityAdaptationTest(unittest.TestCase):
    def test_gpu_scheduler_disables_manual_priority_preemption(self):
        source = Path("research/www_gpu_baselines_20260929/scheduler.py").read_text()
        tree = ast.parse(source)
        scheduler = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef)
                         and node.name == "GPUBaselineScheduler")
        assignments = {
            target.id: node.value.value
            for node in scheduler.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
            if isinstance(node.value, ast.Constant)
        }
        self.assertIs(assignments["manual_priority_preemption"], False)


if __name__ == "__main__":
    unittest.main()
