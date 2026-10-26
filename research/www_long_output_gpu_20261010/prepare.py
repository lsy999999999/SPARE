"""Freeze Conversation requests with outputs beyond the old 512-token cap."""
import csv
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_long_output_gpu_20261010.common import *

MODERATE_DATES = tuple(f"2024-05-{day:02d}" for day in range(12, 18))
HIGH_DATES = ("2024-05-12", "2024-05-14", "2024-05-16")
BANDS = {"output513_1024": (513, 1024, MODERATE_DATES),
         "output1025_1500": (1025, 1500, HIGH_DATES)}


def prior_ids():
    seen = set()
    scanned = []
    for pattern in ("v1*/*inputs/*.json", "www*/*inputs/*.json"):
        for path in sorted((ROOT / "artifacts").glob(pattern)):
            if OUT in path.parents:
                continue
            data = read(path)
            if isinstance(data, list):
                seen.update(str(row["id"]) for row in data
                            if isinstance(row, dict) and "id" in row)
                scanned.append(str(path.relative_to(ROOT)))
    for path in sorted((ROOT / "writing/www/experiments/inputs").glob("*.json")):
        seen.update(str(row["id"]) for row in read(path))
        scanned.append(str(path.relative_to(ROOT)))
    return seen, scanned


def main():
    if (OUT / "design.json").exists():
        raise RuntimeError("design already frozen")
    prior = ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1"
    old = read(prior / "design.json")
    reference = next(cell for cell in old["cells"] if cell["window"] == "conversation_new0")
    history_source = prior / reference["history_file"]
    history_target = OUT / "history/conversation.json"
    history_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(history_source, history_target)

    seen, scanned = prior_ids()
    chosen = {(band, day): [] for band, (_, _, dates) in BANDS.items() for day in dates}
    selected = set()
    trace = ROOT / "data/azure_conv_1week.csv"
    with trace.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for index, row in enumerate(reader):
            timestamp = row[0][:16]
            if timestamp < "2024-05-12 09:00":
                continue
            if all(len(rows) == 64 for rows in chosen.values()):
                break
            day = timestamp[:10]
            if timestamp < f"{day} 09:00":
                continue
            prompt, output = int(row[1]), int(row[2])
            rid = f"conversation:raw:{index}"
            if rid in seen or rid in selected or prompt <= 0 or prompt + output > 8000:
                continue
            for band, (minimum, maximum, dates) in BANDS.items():
                key = (band, day)
                if day in dates and minimum <= output <= maximum and len(chosen[key]) < 64:
                    chosen[key].append(dict(id=rid, prompt=prompt, output=output,
                                            raw_timestamp=row[0]))
                    selected.add(rid)
                    break
    assert all(len(rows) == 64 for rows in chosen.values())
    assert not selected & seen and len(selected) == 64 * len(chosen)

    cells = []
    for (band, day), rows in chosen.items():
        window = f"conversation_{band}_{day.replace('-', '')}"
        raw_path = OUT / "raw_inputs" / f"{window}.json"
        write(raw_path, rows)
        mixed = [dict(id=row["id"], prompt=row["prompt"], output=row["output"],
                      arrival_ms=0. if index < 16 else (index - 15) * 1000 / 5.25)
                 for index, row in enumerate(rows)]
        input_path = OUT / "inputs" / f"{window}_burst_background.json"
        write(input_path, mixed)
        for capacity in (768, 2048, 4096):
            cells.append(dict(key=f"{window}_burst_background_c{capacity}",
                window=window, source="conversation", date=day, output_band=band,
                shape="burst_background", capacity=capacity, budget=1024,
                chunk=256, requests=64, source_input=str(raw_path.relative_to(ROOT)),
                source_input_sha256=digest(raw_path),
                input=str(input_path.relative_to(OUT)), input_sha256=digest(input_path),
                history_file=str(history_target.relative_to(OUT)),
                history_sha256=read(history_target)["source_sha256"],
                history_file_sha256=digest(history_target)))

    for logical, source in source_paths().items():
        target = OUT / "source" / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    design = dict(schema=1, frozen=True, frozen_at=time.time(),
        provenance=dict(trace=str(trace.relative_to(ROOT)), trace_sha256=digest(trace),
            bands={name: [minimum, maximum, list(dates)]
                   for name, (minimum, maximum, dates) in BANDS.items()},
            overlap_audit=dict(prior_ids=len(seen), scanned_files=len(scanned),
                               selected_ids=len(selected), overlap=0)),
        cells=cells, methods=list(ARMS), options=OPTIONS, repeats=2, aa_repeats=2,
        expected_formal_runs=len(cells) * len(ARMS) * 2,
        expected_aa_runs=len(cells) * 2,
        max_model_len=8176, max_num_seqs=64, max_num_batched_tokens=2048,
        block_size=16, dtype="float16", gpu_candidates=[2, 6],
        episode_timeout_s=2400, deadline_hours=96, max_group_retries=2,
        ordering="one fixed shuffled/reverse pair per cell",
        source_hashes=source_hashes(),
        objective="mean TTFT primary; correctness and full service metrics retained",
        inference_unit="six days within one Azure Conversation trace; output bands share dates",
        arrival="first 16 requests at 0 ms; remaining 48 at (index-15)/5.25 s",
        stopping="full frozen matrix; no outcome-based exclusions")
    write(OUT / "design.json", design)
    print(json.dumps(dict(cells=len(cells), formal_runs=design["expected_formal_runs"],
        aa_runs=design["expected_aa_runs"], selected=len(selected), output=str(OUT))))


if __name__ == "__main__":
    main()
