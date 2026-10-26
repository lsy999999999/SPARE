"""Pin the available Qwen3-8B weights without modifying the shared source."""
from __future__ import annotations

import json
from pathlib import Path

from research.www_cross_model_20261021.download_models import BASE, check_weights, sha256


SOURCE = Path("/mnt/disk1/wangshengwei/qwen3-8b")
TARGET = BASE / "qwen3_8b"


def main() -> None:
    index = json.loads((SOURCE / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if not (SOURCE / "config.json").is_file() or not (SOURCE / "tokenizer.json").is_file():
        raise RuntimeError("Qwen3-8B config or tokenizer is missing")
    if not all((SOURCE / shard).is_file() for shard in shards):
        raise RuntimeError("Qwen3-8B weight shard is missing")
    TARGET.mkdir(parents=True, exist_ok=True)
    for path in SOURCE.iterdir():
        if not path.is_file():
            continue
        link = TARGET / path.name
        if link.is_symlink() and link.resolve() == path.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(f"refusing to replace {link}")
        link.symlink_to(path)
    check_weights(TARGET, sorted(TARGET.glob("*.safetensors")))
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(TARGET.iterdir()) if path.is_file()}
    metadata = TARGET / "download.json"
    payload = {"repo": "Qwen/Qwen3-8B", "revision": "shared-local-snapshot",
               "source": str(SOURCE), "files": files}
    if metadata.exists():
        if json.loads(metadata.read_text()) != payload:
            raise RuntimeError("Qwen3-8B source differs from frozen manifest")
    else:
        temporary = metadata.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(metadata)
    print(f"Qwen3-8B ready: {len(shards)} shards, {sum(files[s]['bytes'] for s in shards)} bytes")


if __name__ == "__main__":
    main()
