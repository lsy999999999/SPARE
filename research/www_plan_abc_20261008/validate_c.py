"""Native allocation checks on archived G2 states before C GPU measurement."""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

from research.v1_budget_20260922.policy import State
from research.www_evaluation_gpu_completion_20261003.policy import select as g2_select
from research.www_plan_abc_20261008.c_policy import select

ROOT = Path(__file__).resolve().parents[2]
G2 = ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1"
OUT = ROOT / "artifacts/www_plan_abc_20261008_r1/c"


def read(path):
    return json.loads(path.read_text())


def main():
    design = read(G2 / "design.json")
    counts = Counter()
    for cell in design["cells"]:
        if cell["shape"] != "burst_stop" or cell["capacity"] not in (768, 2048):
            continue
        group = read(G2 / "groups" / f"{cell['key']}.json")
        folder = G2 / group["folder"]
        run = read(folder / "runs/arrival_r0.json")
        with gzip.open(folder / run["rounds_path"], "rt") as handle:
            rounds = json.load(handle)
        for row in rounds:
            states = [State(**state) for state in row["state"]]
            args = (states, 1024, 256, row["free_before"],
                    row["total_blocks"], 16, 32, 8176)
            threshold = run["config"]["threshold"]
            original = g2_select(*args, threshold=threshold, recent=row["features"])
            assert original["threshold"] == row["threshold"]
            if original["selected"] is not None:
                assert original["selected"].tokens == row["scheduled"]
            reproduced = select(*args, threshold=threshold, recent=row["features"])
            assert (reproduced["budget"], reproduced["chunk"]) == (original["budget"], original["chunk"])
            assert reproduced["selected"] == original["selected"]
            counts["g2_rounds"] += 1
            for budget in (128, 256, 384, 512, 768, 1024):
                choice = select(*args, threshold=threshold, recent=row["features"],
                    options=dict(mode="fixed_guarded", fixed_budget=budget))
                counts["guarded_choices"] += 1
                if choice["base"] is None:
                    assert choice["selected"] is None
                    counts["recovery_fallback"] += 1
                    continue
                assert choice["selected"] is not None
                if choice["changed"]:
                    assert choice["candidate_diagnostics"][0]["guard"]
                    assert choice["selected"].growth == choice["candidate_diagnostics"][0]["growth"]
                    for state in states:
                        if state.rid not in choice["base"].tokens:
                            continue
                        protected = state.decode or (state.outputs == 0
                            and choice["base"].tokens[state.rid] == state.remaining)
                        if protected:
                            assert choice["selected"].tokens.get(state.rid, 0) >= choice["base"].tokens[state.rid]
                    counts["guarded_changes"] += 1
                else:
                    counts["guarded_fallbacks"] += 1
    assert counts["g2_rounds"] > 0 and counts["guarded_changes"] > 0
    result = dict(passed=True, checks=dict(counts),
        scope="12 archived G2 burst-stop trajectories, native allocator, all six fixed candidate budgets")
    path = OUT / "validation.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
