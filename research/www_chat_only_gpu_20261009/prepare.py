"""Select six unused Conversation windows and freeze all GPU cells."""
import csv
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.www_chat_only_gpu_20261009.common import *

ANCHORS = tuple(f"2024-05-{day:02d} 09:00" for day in range(12, 18))


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
    chosen = {anchor: [] for anchor in ANCHORS}
    selected = set()
    trace = ROOT / "data/azure_conv_1week.csv"
    with trace.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for index, row in enumerate(reader):
            timestamp = row[0][:16]
            if timestamp < ANCHORS[0]:
                continue
            if all(len(rows) == 128 for rows in chosen.values()):
                break
            date = timestamp[:10]
            anchor = f"{date} 09:00"
            if anchor not in chosen or timestamp < anchor or len(chosen[anchor]) == 128:
                continue
            prompt, output = int(row[1]), int(row[2])
            rid = f"conversation:raw:{index}"
            if rid in seen or rid in selected or not (prompt > 0 and 0 < output <= 512
                                                     and prompt + output <= 8000):
                continue
            chosen[anchor].append(dict(id=rid, prompt=prompt, output=output,
                                       raw_timestamp=row[0]))
            selected.add(rid)
    assert all(len(rows) == 128 for rows in chosen.values())
    assert not selected & seen and len(selected) == 768

    cells = []
    for number, anchor in enumerate(ANCHORS):
        window = f"conversation_chat{number}"
        raw_path = OUT / "raw_inputs" / f"{window}.json"
        write(raw_path, chosen[anchor])
        mixed = [dict(id=row["id"], prompt=row["prompt"], output=row["output"],
                      arrival_ms=0. if index < 32 else (index - 31) * 1000 / 5.25)
                 for index, row in enumerate(chosen[anchor])]
        input_path = OUT / "inputs" / f"{window}_burst_background.json"
        write(input_path, mixed)
        for capacity in (768, 2048, 4096):
            cells.append(dict(key=f"{window}_burst_background_c{capacity}",
                window=window, source="conversation", anchor=anchor,
                shape="burst_background", capacity=capacity, budget=1024,
                chunk=256, requests=128, source_input=str(raw_path.relative_to(ROOT)),
                source_input_sha256=digest(raw_path),
                input=str(input_path.relative_to(OUT)), input_sha256=digest(input_path),
                history_file=str(history_target.relative_to(OUT)),
                history_sha256=read(history_target)["source_sha256"],
                history_file_sha256=digest(history_target)))

    for logical, source in source_paths().items():
        target = OUT / "source" / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    hashes = source_hashes()
    design = dict(schema=1, frozen=True, frozen_at=time.time(),
        provenance=dict(trace=str(trace.relative_to(ROOT)), trace_sha256=digest(trace),
            anchors=ANCHORS, overlap_audit=dict(prior_ids=len(seen),
                scanned_files=len(scanned), selected_ids=len(selected), overlap=0)),
        cells=cells, methods=list(ARMS), options=OPTIONS, repeats=4, aa_repeats=2,
        expected_formal_runs=len(cells) * len(ARMS) * 4,
        expected_aa_runs=len(cells) * 2,
        max_model_len=8176, max_num_seqs=64, max_num_batched_tokens=2048,
        block_size=16, dtype="float16", gpu_candidates=[2, 6],
        episode_timeout_s=900, deadline_hours=72, max_group_retries=2,
        ordering="two fixed shuffled/reverse pairs per cell",
        source_hashes=hashes,
        objective="mean TTFT primary; all results retained including regressions",
        inference_unit="six newly selected Conversation windows from one Azure trace",
        arrival="first 32 requests at 0 ms; remaining 96 at (index-31)/5.25 s",
        stopping="full frozen matrix; no outcome-based exclusions")
    write(OUT / "design.json", design)
    print(json.dumps(dict(cells=len(cells), formal_runs=design["expected_formal_runs"],
        aa_runs=design["expected_aa_runs"], prior_ids=len(seen), output=str(OUT))))


if __name__ == "__main__":
    main()
