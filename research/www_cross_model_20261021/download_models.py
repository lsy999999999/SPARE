"""Download and inventory the two frozen cross-model GPU candidates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time


BASE = Path("models")
MODELS = {
    "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_status(state: str, model: str, detail: str = "") -> None:
    target = BASE.parent / "download_status.json"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps({"state": state, "model": model,
                                     "detail": detail, "updated": time.time()},
                                    indent=2) + "\n")
    temporary.replace(target)


def fetch(url: str, target: Path) -> None:
    if target.exists():
        return
    temporary = target.with_name(target.name + ".partial")
    for attempt in range(5):
        result = subprocess.run(
            ["curl", "--location", "--fail", "--silent", "--show-error",
             "--retry", "3", "--retry-all-errors", "--connect-timeout", "20",
             "--max-time", "7200", "--continue-at", "-", "--output",
             str(temporary), url], check=False)
        if result.returncode == 0:
            temporary.replace(target)
            return
        if attempt == 4:
            raise RuntimeError(f"download failed: {url}")
        time.sleep(2 ** attempt)


def check_weights(target: Path, weights: list[Path]) -> None:
    index = target / "model.safetensors.index.json"
    manifest = json.loads(index.read_text()) if index.exists() else None
    weight_map = manifest["weight_map"] if manifest else None
    if weight_map is not None and set(weight_map.values()) != {p.name for p in weights}:
        raise RuntimeError(f"weight shard list mismatch: {target.name}")

    observed_keys = set()
    data_bytes = 0
    for path in weights:
        with path.open("rb") as stream:
            header_bytes = stream.read(8)
            if len(header_bytes) != 8:
                raise RuntimeError(f"truncated safetensors header: {path}")
            header_size = struct.unpack("<Q", header_bytes)[0]
            if header_size > 100 * 1024 * 1024:
                raise RuntimeError(f"invalid safetensors header size: {path}")
            header = json.loads(stream.read(header_size))
        entries = {key: value for key, value in header.items() if key != "__metadata__"}
        if not entries:
            raise RuntimeError(f"empty safetensors shard: {path}")
        ends = []
        for key, value in entries.items():
            start, end = value["data_offsets"]
            if start < 0 or end <= start:
                raise RuntimeError(f"invalid tensor offset: {path}: {key}")
            if key in observed_keys or (weight_map is not None and weight_map.get(key) != path.name):
                raise RuntimeError(f"tensor mapping mismatch: {path}: {key}")
            observed_keys.add(key)
            ends.append(end)
            data_bytes += end - start
        if path.stat().st_size != 8 + header_size + max(ends):
            raise RuntimeError(f"safetensors file size mismatch: {path}")
    if weight_map is not None:
        if observed_keys != set(weight_map):
            raise RuntimeError(f"tensor key list mismatch: {target.name}")
        expected = manifest.get("metadata", {}).get("total_size")
        if expected is not None and data_bytes != expected:
            raise RuntimeError(f"tensor data size mismatch: {target.name}: {data_bytes}/{expected}")


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    for name, repo in MODELS.items():
        target = BASE / name
        target.mkdir(parents=True, exist_ok=True)
        metadata = target / "download.json"
        if metadata.exists():
            print(name, "already complete", flush=True)
            continue
        write_status("running", name)
        print(name, "resolving", repo, flush=True)
        metadata_json = target / "model_info.json"
        fetch(f"https://hf-mirror.com/api/models/{repo}", metadata_json)
        info = json.loads(metadata_json.read_text())
        revision = info["sha"]
        print(name, "revision", revision, flush=True)
        files = sorted(
            item["rfilename"] for item in info["siblings"]
            if "/" not in item["rfilename"] and (
                item["rfilename"].endswith((".json", ".safetensors"))
                or item["rfilename"].startswith("tokenizer.model")
                or item["rfilename"] in ("vocab.json", "merges.txt")))
        if "model.safetensors.index.json" in files:
            fetch(f"https://hf-mirror.com/{repo}/resolve/{revision}/model.safetensors.index.json",
                  target / "model.safetensors.index.json")
            shards = set(json.loads((target / "model.safetensors.index.json").read_text())["weight_map"].values())
            files = [filename for filename in files
                     if not filename.endswith(".safetensors") or filename in shards]
        for filename in files:
            print(name, filename, "downloading", flush=True)
            fetch(f"https://hf-mirror.com/{repo}/resolve/{revision}/{filename}",
                  target / filename)
        weights = sorted(target.glob("*.safetensors"))
        if not (target / "config.json").is_file() or not weights:
            raise RuntimeError(f"incomplete model download: {name}")
        check_weights(target, weights)
        files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(target.iterdir()) if p.is_file()}
        metadata.write_text(json.dumps({"repo": repo, "revision": revision,
                                        "files": files}, indent=2) + "\n")
        print(name, "complete", sum(p.stat().st_size for p in weights), flush=True)
        write_status("complete", name)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if BASE.exists():
            write_status("failed", "unknown", str(error))
        raise
