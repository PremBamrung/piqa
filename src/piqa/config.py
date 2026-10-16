"""Settings from `piqa.toml`, searched from the current directory upwards.

Example (paths are relative to the toml file):

    [dinov3]
    repo = "dinov3_repo"     # local clone of facebookresearch/dinov3
    weights = "dinov3"       # a .pth file, or a folder holding <model>_pretrain_*.pth
    model = "dinov3_vits16"

Environment variables PIQA_DINOV3_REPO / PIQA_DINOV3_WEIGHTS / PIQA_DINOV3_MODEL override it.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_NAME = "piqa.toml"
DEFAULT_MODEL = "dinov3_vits16"


@dataclass
class Dinov3Config:
    repo: Path | None
    weights: Path | None
    model: str

    def resolve_weights(self) -> Path:
        """The checkpoint file, picking `<model>_pretrain_*.pth` when `weights` is a folder."""
        if self.weights is None:
            raise SystemExit(f"DINOv3 weights not set: add [dinov3] weights to {CONFIG_NAME}")
        if self.weights.is_dir():
            matches = sorted(self.weights.glob(f"{self.model}_pretrain_*.pth"))
            if not matches:
                raise SystemExit(f"No {self.model}_pretrain_*.pth in {self.weights}")
            return matches[0]
        if not self.weights.exists():
            raise SystemExit(f"DINOv3 weights not found: {self.weights}")
        return self.weights

    def resolve_repo(self) -> Path:
        if self.repo is None or not (self.repo / "hubconf.py").exists():
            raise SystemExit(
                f"DINOv3 repo not found ({self.repo}): set [dinov3] repo in {CONFIG_NAME}"
            )
        return self.repo


def find_config(start: Path | None = None) -> Path | None:
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        candidate = folder / CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


def load_dinov3_config() -> Dinov3Config:
    section: dict = {}
    base = Path.cwd()
    path = find_config()
    if path is not None:
        section = tomllib.loads(path.read_text(encoding="utf-8")).get("dinov3", {})
        base = path.parent

    def as_path(env: str, key: str) -> Path | None:
        value = os.environ.get(env) or section.get(key)
        return (base / value).resolve() if value else None

    return Dinov3Config(
        repo=as_path("PIQA_DINOV3_REPO", "repo"),
        weights=as_path("PIQA_DINOV3_WEIGHTS", "weights"),
        model=os.environ.get("PIQA_DINOV3_MODEL") or section.get("model", DEFAULT_MODEL),
    )
