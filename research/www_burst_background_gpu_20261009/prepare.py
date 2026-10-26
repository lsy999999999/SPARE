"""Freeze the full mixed-arrival GPU matrix before running it."""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_burst_background_gpu_20261009.common import *


def main():
    if (OUT / "design.json").exists():
        raise RuntimeError("design already frozen")
    prior = ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1"
    old = read(prior / "design.json")
    cells = []
    for window in ("code_new0", "code_new1", "code_new2",
                   "conversation_new0", "conversation_new1", "conversation_new2"):
        source = prior / "inputs" / f"{window}_burst_stop.json"
        rows = read(source)
        assert len(rows) == 128 and all(row["arrival_ms"] == 0 for row in rows)
        mixed = [dict(row, arrival_ms=0. if index < 32 else
                      (index - 31) * 1000 / 5.25)
                 for index, row in enumerate(rows)]
        target = OUT / "inputs" / f"{window}_burst_background.json"
        write(target, mixed)
        previous = next(cell for cell in old["cells"] if cell["window"] == window)
        history_source = prior / previous["history_file"]
        history_target = OUT / "history" / f"{window.split('_')[0]}.json"
        history_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(history_source, history_target)
        for capacity in (768, 2048, 4096):
            cells.append(dict(key=f"{window}_burst_background_c{capacity}",
                window=window, shape="burst_background",
                source_input=str(source.relative_to(ROOT)),
                source_input_sha256=digest(source),
                input=str(target.relative_to(OUT)), input_sha256=digest(target),
                history_file=str(history_target.relative_to(OUT)),
                history_sha256=read(history_target)["source_sha256"],
                history_file_sha256=digest(history_target),
                capacity=capacity, budget=1024, chunk=256, requests=128))
    for logical, source in source_paths().items():
        target = OUT / "source" / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    hashes = source_hashes()
    design = dict(schema=1, frozen=True, frozen_at=time.time(),
        cells=cells, methods=list(ARMS), options=OPTIONS, repeats=4, aa_repeats=2,
        expected_formal_runs=len(cells) * len(ARMS) * 4,
        expected_aa_runs=len(cells) * 2,
        max_model_len=8176, max_num_seqs=64, max_num_batched_tokens=2048,
        block_size=16, dtype="float16", gpu_candidates=[2, 6],
        episode_timeout_s=900, deadline_hours=72, max_group_retries=2,
        ordering="two fixed shuffled/reverse pairs per cell",
        source_hashes=hashes,
        objective="mean TTFT primary; full positive and negative results retained",
        inference_unit="six reused source windows; capacities paired within windows",
        arrival="first 32 requests at 0 ms; remaining 96 at (index-31)/5.25 s",
        stopping="full frozen matrix; no outcome-based exclusions")
    write(OUT / "design.json", design)
    print(json.dumps(dict(cells=len(cells), formal_runs=design["expected_formal_runs"],
                          aa_runs=design["expected_aa_runs"], output=str(OUT))))


if __name__ == "__main__":
    main()
