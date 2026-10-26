"""Shared constants and immutable file helpers for the WWW GPU baseline matrix."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / os.environ.get(
    "WWW_GPU_OUT", "artifacts/www_gpu_baselines_20260929_r2")
MODEL = Path("models/Qwen2.5-7B-Instruct")
PYTHON = Path(".venv/bin/python")

ARMS = (
    "arrival", "basic", "srp", "native", "sarathi", "past_future",
    "ssjf_oracle", "mc_sf_oracle", "trail_oracle", "tie_oracle_proxy",
    "uniboost_static", "apt_serve_kv_rule", "sola_rule",
)
ORACLE_ARMS = (
    "ssjf_oracle", "mc_sf_oracle", "trail_oracle", "tie_oracle_proxy",
)
METHOD = {
    "arrival": "threshold", "basic": "threshold", "srp": "srp",
    "native": "native", "sarathi": "sarathi_v1",
    "past_future": "past_future_v1", "ssjf_oracle": "ssjf_oracle",
    "mc_sf_oracle": "mc_sf_oracle", "trail_oracle": "trail_oracle",
    "tie_oracle_proxy": "tie_oracle_proxy",
    "uniboost_static": "uniboost_static",
    "apt_serve_kv_rule": "apt_serve_kv_rule", "sola_rule": "sola_rule",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def source_paths():
    paths = {str(path.relative_to(ROOT)): path for path in HERE.glob("*.py")}
    protocol = HERE / "PROTOCOL_CN.md"
    if protocol.exists():
        paths[str(protocol.relative_to(ROOT))] = protocol
    for folder, names in (
        ("v1_budget_20260922", ("policy.py", "scheduler.py")),
        ("v1_saving_threshold_20260924", ("policy.py",)),
        ("v1_broad12h_20260925", ("policy.py", "scheduler.py")),
        ("v1_paper_comparison_20260922", ("rules.py", "scheduler.py")),
    ):
        for name in names:
            logical = Path("research") / folder / name
            snapshot = OUT / "source" / logical
            paths[str(logical)] = snapshot if snapshot.exists() else ROOT / logical
    return paths


def source_hashes():
    return {logical: digest(path) for logical, path in sorted(source_paths().items())}
