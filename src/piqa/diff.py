"""Files present under folder A but missing (same relative path) under folder B."""

from __future__ import annotations

import os
from pathlib import Path


def missing(a: Path, b: Path) -> list[str]:
    result = []
    for dirpath, _, filenames in os.walk(a):
        for name in filenames:
            rel = (Path(dirpath) / name).relative_to(a)
            if not (b / rel).exists():
                result.append(rel.as_posix())
    return sorted(result)


def diff(a: Path, b: Path, *, verbose: bool = False) -> None:
    for folder in (a, b):
        if not folder.is_dir():
            raise SystemExit(f"Not a folder: {folder}")
    files = missing(a, b)
    if verbose:
        for rel in files:
            print(rel)
    print(f"[piqa] {len(files)} files in {a} are missing from {b}")
