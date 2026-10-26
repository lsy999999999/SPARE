"""Select spaced, nonoverlapping request windows from the original Azure traces."""
from __future__ import annotations

import csv
from collections import Counter

from research.www_evaluation_completion_20261003.common import ROOT, read, write, digest


ANCHORS = {
    "code": ["2024-05-11 16:37", "2024-05-13 16:37", "2024-05-16 16:37"],
    "conversation": ["2024-05-13 16:37", "2024-05-15 16:37", "2024-05-18 16:37"],
}
SAMPLE_COUNT = 10
WINDOW_SIZE = 512


def eligible(row: list[str], anchor: str) -> bool:
    if row[0][:16] < anchor or row[0][:10] != anchor[:10]:
        return False
    prompt, output = int(row[1]), int(row[2])
    return prompt > 0 and 0 < output <= 512 and prompt + output <= 8000


def extract(out) -> dict:
    manifest_path = out / "samples.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        for item in manifest["samples"]:
            if digest(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"Sample changed: {item['path']}")
        return manifest
    counts = Counter()
    sources = {name: ROOT / f"data/azure_{'conv' if name == 'conversation' else 'code'}_1week.csv"
               for name in ANCHORS}
    for source, path in sources.items():
        anchors = ANCHORS[source]
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for row in reader:
                stamp = row[0][:16]
                for index, anchor in enumerate(anchors):
                    if stamp >= anchor and row[0][:10] == anchor[:10] and eligible(row, anchor):
                        counts[source, index] += 1
    starts = {}
    for key, total in counts.items():
        if total < SAMPLE_COUNT * WINDOW_SIZE:
            raise RuntimeError(f"Too few eligible requests for {key}: {total}")
        starts[key] = [i * (total - WINDOW_SIZE) // (SAMPLE_COUNT - 1)
                       for i in range(SAMPLE_COUNT)]
        assert all(b - a >= WINDOW_SIZE for a, b in zip(starts[key], starts[key][1:]))
    selected = {(source, index, sample): [] for source in ANCHORS
                for index in range(3) for sample in range(SAMPLE_COUNT)}
    for source, path in sources.items():
        anchors = ANCHORS[source]
        seen = Counter()
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for raw_index, row in enumerate(reader):
                for index, anchor in enumerate(anchors):
                    if not eligible(row, anchor):
                        continue
                    position = seen[source, index]
                    seen[source, index] += 1
                    for sample, start in enumerate(starts[source, index]):
                        if start <= position < start + WINDOW_SIZE:
                            selected[source, index, sample].append(dict(
                                id=f"{source}:raw:{raw_index}", prompt=int(row[1]),
                                output=int(row[2]), raw_timestamp=row[0]))
    items = []
    for (source, index, sample), rows in sorted(selected.items()):
        assert len(rows) == WINDOW_SIZE
        window = f"{source}_new{index}"
        path = out / "samples" / f"{window}_s{sample:02d}.json"
        write(path, rows)
        items.append(dict(window=window, sample=sample, path=str(path.relative_to(ROOT)),
                          sha256=digest(path), first_id=rows[0]["id"], last_id=rows[-1]["id"],
                          first_timestamp=rows[0]["raw_timestamp"],
                          last_timestamp=rows[-1]["raw_timestamp"]))
    manifest = dict(sample_count=SAMPLE_COUNT, window_size=WINDOW_SIZE,
                    selection="Eligible requests after each fixed date anchor; 10 evenly spaced, "
                              "nonoverlapping contiguous 512-request segments per date.",
                    source_hashes={str(path.relative_to(ROOT)): digest(path)
                                   for path in sources.values()}, samples=items)
    write(manifest_path, manifest)
    return manifest
