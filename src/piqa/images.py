"""Image discovery, fast decoding and the per-folder `.piqa/` thumbnail cache."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
CACHE_DIRNAME = ".piqa"

# Thumbnail tiers, largest first: "m" for previews/lightbox, "s" for the wall.
THUMB_TIERS = {"m": 512, "s": 128}


def is_junk(name: str) -> bool:
    """macOS AppleDouble forks (`._DSCF0001.JPG`) and other hidden files."""
    return name.startswith(".")


def find_images(root: Path) -> list[Path]:
    """All supported images under `root`, sorted, skipping hidden files and dirs (incl. `.piqa`)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not is_junk(d)]
        for name in filenames:
            if not is_junk(name) and Path(name).suffix.lower() in IMAGE_EXTS:
                found.append(Path(dirpath) / name)
    return sorted(found)


def cache_dir(root: Path) -> Path:
    return root / CACHE_DIRNAME


def fingerprint(path: Path) -> str:
    """Cheap change detector; whole seconds so it matches across Windows, WSL and macOS."""
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def open_image(path: Path, min_size: int | None = None) -> Image.Image:
    """Open upright RGB. For JPEGs, decode at the smallest DCT scale still >= `min_size`.

    A 26 MP JPEG decoded at 1/8 scale is ~780 px and many times faster than a full decode.
    """
    img = Image.open(path)
    if min_size and img.format == "JPEG":
        img.draft("RGB", (min_size, min_size))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def thumb_path(cache: Path, rel: str, tier: str) -> Path:
    return cache / "thumbs" / tier / f"{rel}.jpg"


def has_thumbs(cache: Path, rel: str) -> bool:
    return all(thumb_path(cache, rel, tier).exists() for tier in THUMB_TIERS)


def write_thumbs(img: Image.Image, cache: Path, rel: str) -> None:
    """Write every thumbnail tier for `rel`, downscaling progressively from the largest."""
    thumb = img.copy()
    for tier, size in THUMB_TIERS.items():
        thumb.thumbnail((size, size), Image.LANCZOS)
        path = thumb_path(cache, rel, tier)
        path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(path, format="JPEG", quality=80)
