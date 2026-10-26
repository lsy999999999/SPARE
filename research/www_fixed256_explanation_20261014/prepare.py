"""Freeze the earliest existing long-output window in each source and band."""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_fixed256_explanation_20261014.common import *


def main():
    if (OUT / "design.json").exists():
        raise RuntimeError("design already frozen")
    cells = []
    provenance = {}
    for source, prior_name in (
        ("code", "www_code_long_output_gpu_20261010_r1"),
        ("conversation", "www_long_output_gpu_20261010_r1"),
    ):
        prior = ROOT / "artifacts" / prior_name
        design = read(prior / "design.json")
        provenance[source] = dict(design=str((prior / "design.json").relative_to(ROOT)),
                                  sha256=digest(prior / "design.json"))
        for band in ("output513_1024", "output1025_1500"):
            options = [cell for cell in design["cells"]
                       if cell["output_band"] == band and cell["capacity"] == 768]
            first = min(options, key=lambda cell: (cell["date"], cell["window"]))
            input_source = prior / first["input"]
            target = OUT / "inputs" / Path(first["input"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_source, target)
            if digest(target) != first["input_sha256"]:
                raise RuntimeError(f"input mismatch: {input_source}")
            history_source = prior / first["history_file"]
            history_target = OUT / "history" / f"{source}.json"
            history_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(history_source, history_target)
            for capacity in (768, 2048, 4096):
                cells.append(dict(key=f"{first['window']}_burst_background_c{capacity}",
                    window=first["window"], source=source, date=first["date"],
                    output_band=band, shape="burst_background", capacity=capacity,
                    budget=1024, chunk=256, requests=64,
                    input=str(target.relative_to(OUT)), input_sha256=digest(target),
                    history_file=str(history_target.relative_to(OUT)),
                    history_sha256=read(history_target)["source_sha256"],
                    history_file_sha256=digest(history_target)))
    if len(cells) != 12 or len({c["key"] for c in cells}) != 12:
        raise RuntimeError("unexpected pilot matrix")
    for logical, source in source_paths().items():
        target = OUT / "source" / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    design = dict(schema=1, frozen=True, frozen_at=time.time(),
        provenance=provenance, cells=cells, methods=list(ARMS), options=OPTIONS,
        repeats=4, aa_repeats=2, bookend_repeats=2,
        expected_formal_runs=len(cells) * len(ARMS) * 4,
        expected_aa_runs=len(cells) * 2, expected_bookend_runs=len(cells) * 2,
        max_model_len=8176, max_num_seqs=64, max_num_batched_tokens=2048,
        block_size=16, dtype="float16", gpu_candidates=[2, 6],
        episode_timeout_s=21600, deadline_hours=120, max_group_retries=2,
        ordering="four paired repeats; shuffled order then reverse order",
        source_hashes=source_hashes(),
        objective="explain full versus fixed256 using decisions, resource use and TTFT",
        inference_unit="one frozen window per source and output band",
        arrival="first 16 requests at 0 ms; remaining 48 at (index-15)/5.25 s",
        stopping="complete frozen matrix; failures and timeouts retained")
    write(OUT / "design.json", design)
    print(f"frozen {len(cells)} cells, {design['expected_formal_runs']} formal runs")


if __name__ == "__main__":
    main()
