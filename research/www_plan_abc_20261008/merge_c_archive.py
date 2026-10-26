"""Merge the remotely produced C artifact without overwriting local files."""
from __future__ import annotations

import hashlib
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = Path("artifacts/www_plan_abc_20261008_r1/c")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main(archive: Path) -> None:
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for item in members:
            path = Path(item.name)
            if not path.is_relative_to(PREFIX) or not (item.isfile() or item.isdir()):
                raise ValueError(f"Unexpected archive member: {item.name}")
        for item in members:
            target = ROOT / item.name
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(item)
            assert source is not None
            if target.exists():
                remote_hash = hashlib.sha256()
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    remote_hash.update(chunk)
                if digest(target) != remote_hash.hexdigest():
                    raise ValueError(f"Local and remote files differ: {target}")
            else:
                with target.open("xb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
    print(f"Merged {len(members)} C archive members without overwriting local files")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
