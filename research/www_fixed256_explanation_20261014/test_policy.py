"""Check the experimental Full arm against the frozen evaluation selector."""
import gzip
import json
import unittest
from pathlib import Path

from research.v1_budget_20260922.policy import State
from research.www_evaluation_gpu_completion_20261003.policy import select as old_select
from research.www_fixed256_explanation_20261014.policy import select


ROOT = Path(__file__).resolve().parents[2]


class PolicyParityTest(unittest.TestCase):
    def test_archived_round_states(self):
        files = sorted((ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1")
                       .glob("attempts/*/a*/rounds/arrival_r0.json.gz"))
        self.assertTrue(files, "archived GPU round traces required")
        checked = 0
        for path in files[:4]:
            with gzip.open(path, "rt") as handle:
                rounds = json.load(handle)
            for row in rounds[::max(1, len(rounds) // 200)]:
                states = [State(**item) for item in row["state"]]
                args = (states, 1024, 256, row["free_before"],
                        row["total_blocks"], 16, 32, 8176)
                recent = {"arrival_rate": row["features"]["arrival_rate"]}
                old = old_select(*args, recent=recent)
                new = select(*args, recent=recent)
                for key in ("budget", "chunk", "candidates", "changed",
                            "rejected_guard", "rejected_saving",
                            "rejected_infeasible", "threshold"):
                    self.assertEqual(new[key], old[key], (path, row["round"], key))
                self.assertEqual(new["selected"], old["selected"])
                self.assertEqual(new["candidate_diagnostics"],
                                 old["candidate_diagnostics"])
                for experimental, original in (
                    ({"mode": "fixed"}, {"mode": "fixed", "fixed_budget": 256}),
                    ({"mode": "no_finish"}, {"protect_finish": False}),
                ):
                    changed = select(*args, recent=recent, options=experimental)
                    reference = old_select(*args, recent=recent, options=original)
                    self.assertEqual(changed["budget"], reference["budget"])
                    self.assertEqual(changed["chunk"], reference["chunk"])
                    self.assertEqual(changed["selected"], reference["selected"])
                checked += 1
        self.assertGreaterEqual(checked, 100)


if __name__ == "__main__":
    unittest.main()
