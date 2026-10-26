"""Fit model-specific GPU iteration timing on pilot windows only."""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import nnls

from research.www_evaluation_completion_20261003.timing import NAMES, batch_row


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/www_cross_model_20261021_r1"
OUT = ROOT / "artifacts/www_cross_model_timing_20261022_r1"
MODELS = ("qwen25_3b", "qwen3_8b", "mistral_7b")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_rows() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((SOURCE / "runs").glob("*.json")):
        row = json.loads(path.read_text())
        if row["model"] not in MODELS:
            continue
        if row["status"] != "complete":
            raise RuntimeError(f"incomplete GPU run: {path}")
        row["record_path"] = str(path.relative_to(SOURCE))
        groups[row["model"]].append(row)
    return groups


def rounds(row: dict):
    path = SOURCE / row["round_path"]
    if digest(path) != row["round_sha256"]:
        raise RuntimeError(f"round hash mismatch: {path}")
    with gzip.open(path, "rt") as stream:
        yield from json.load(stream)


def observations(rows: list[dict]):
    for run in rows:
        for step in rounds(run):
            if step["allocations"]:
                elapsed = step["step_ms"]
                if elapsed <= 0:
                    raise RuntimeError(f"nonpositive GPU step: {run['record_path']}")
                yield batch_row(step), elapsed, run


def fit(model: str, rows: list[dict]) -> dict:
    train = [r for r in rows if r["sample"].endswith("d1_w08")]
    heldout = [r for r in rows if not r["sample"].endswith("d1_w08")]
    if not train or not heldout or {r["sample"] for r in train} & {r["sample"] for r in heldout}:
        raise RuntimeError(f"missing disjoint training and holdout rows: {model}")
    x, y = [], []
    for features, elapsed, _ in observations(train):
        x.append(features)
        y.append(elapsed)
    matrix, labels = np.asarray(x), np.asarray(y)
    coefficients, _ = nnls(matrix, labels)
    training_predictions = np.maximum(.1, matrix @ coefficients)
    profile = dict(
        model=model, kind="nonnegative_batch_affine", features=NAMES,
        coefficients=coefficients.tolist(), training_episodes=len(train),
        training_iterations=len(labels),
        training_rmse_ms=float(np.sqrt(np.mean((training_predictions-labels)**2))),
        training_mape_pct=float(100*np.mean(np.abs(training_predictions-labels)/labels)),
        training_samples=sorted({r["sample"] for r in train}),
        holdout_samples=sorted({r["sample"] for r in heldout}),
        training_round_hashes={r["round_path"]: r["round_sha256"] for r in train},
        source="4090 engine.step wall time, including scheduler and instrumentation",
    )
    write(OUT / "profiles" / f"{model}.json", profile)

    episode_errors = []
    for run in heldout:
        predicted, observed = [], []
        for features, elapsed, _ in observations([run]):
            predicted.append(max(.1, float(np.dot(features, coefficients))))
            observed.append(elapsed)
        episode_errors.append(dict(
            model=model, sample=run["sample"], kv_blocks=run["kv_blocks"],
            arm=run["arm"], rate_qps=run["rate_qps"], iterations=len(observed),
            step_mae_ms=statistics.fmean(abs(a-b) for a,b in zip(predicted,observed)),
            step_mape_pct=100*statistics.fmean(abs(a-b)/b for a,b in zip(predicted,observed)),
            total_time_ape_pct=100*abs(sum(predicted)-sum(observed))/sum(observed),
            round_sha256=run["round_sha256"],
        ))
    write(OUT / "holdout" / f"{model}.json", episode_errors)
    return dict(model=model, training_episodes=len(train),
                training_iterations=len(labels), heldout_episodes=len(heldout),
                holdout_step_mae_ms=statistics.fmean(e["step_mae_ms"] for e in episode_errors),
                holdout_step_mape_pct=statistics.fmean(e["step_mape_pct"] for e in episode_errors),
                holdout_total_time_ape_pct=statistics.fmean(e["total_time_ape_pct"] for e in episode_errors))


def main() -> None:
    groups = load_rows()
    summary = [fit(model, groups[model]) for model in MODELS]
    write(OUT / "summary.json", summary)
    for row in summary:
        print(row, flush=True)


if __name__ == "__main__":
    main()
