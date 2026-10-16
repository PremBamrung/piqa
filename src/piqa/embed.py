"""DINOv3 [CLS] embeddings for every image in a folder, cached in `.piqa/embeddings-<model>.npz`.

Each image is read once: the same reduced-scale decode feeds the model and the thumbnail cache.
Re-runs only process new or changed files.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from piqa.config import Dinov3Config
from piqa.images import (
    THUMB_TIERS,
    cache_dir,
    find_images,
    fingerprint,
    has_thumbs,
    open_image,
    write_thumbs,
)


class PadToSquare:
    """Pad the short side with black so aspect ratio doesn't leak into the embedding."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        square = Image.new("RGB", (side, side))
        square.paste(img, ((side - w) // 2, (side - h) // 2))
        return square


def make_transform(img_size: int) -> v2.Compose:
    return v2.Compose(
        [
            PadToSquare(),
            v2.ToImage(),
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class _ImageDataset(Dataset):
    def __init__(self, root: Path, rels: list[str], img_size: int):
        self.root = root
        self.cache = cache_dir(root)
        self.rels = rels
        self.img_size = img_size
        self.decode_size = max(img_size, *THUMB_TIERS.values())
        self.transform = make_transform(img_size)

    def __len__(self) -> int:
        return len(self.rels)

    def __getitem__(self, i: int):
        rel = self.rels[i]
        try:
            img = open_image(self.root / rel, min_size=self.decode_size)
            write_thumbs(img, self.cache, rel)
            return self.transform(img), i, True
        except Exception as exc:
            print(f"\n[WARN] skipping {rel}: {exc}", file=sys.stderr)
            return torch.zeros(3, self.img_size, self.img_size), i, False


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(cfg: Dinov3Config, device: torch.device, half: bool) -> torch.nn.Module:
    # Import the backbones directly: the repo's hubconf.py also imports the segmentation and
    # text heads, which pull in torchmetrics and friends.
    repo = str(cfg.resolve_repo())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from dinov3.hub import backbones

    build = getattr(backbones, cfg.model, None)
    if build is None:
        raise SystemExit(f"Unknown DINOv3 model {cfg.model!r}")
    model = build(weights=str(cfg.resolve_weights()))
    model = model.eval().to(device)
    return model.half() if half else model


def embeddings_path(root: Path, model: str) -> Path:
    return cache_dir(root) / f"embeddings-{model}.npz"


def load_cached(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    """rel path -> (fingerprint, vector) from a previous run, if any."""
    if not path.exists():
        return {}
    data = np.load(path)
    return {
        rel: (fp, vec)
        for rel, fp, vec in zip(
            data["paths"].tolist(), data["fingerprints"].tolist(), data["embeddings"], strict=True
        )
    }


def embed(
    root: Path,
    cfg: Dinov3Config,
    *,
    batch_size: int = 64,
    num_workers: int | None = None,
    img_size: int = 256,
    max_images: int | None = None,
    fp32: bool = False,
) -> Path:
    """Embed every image under `root` and return the path of the embeddings file."""
    paths = find_images(root)
    if max_images:
        paths = paths[:max_images]
    if not paths:
        raise SystemExit(f"No images found under {root}")

    cache = cache_dir(root)
    cache.mkdir(exist_ok=True)
    rels = [p.relative_to(root).as_posix() for p in paths]
    fingerprints = [fingerprint(p) for p in paths]
    out = embeddings_path(root, cfg.model)

    cached = load_cached(out)
    vectors: dict[str, np.ndarray] = {}
    todo = []
    for rel, fp in zip(rels, fingerprints, strict=True):
        hit = cached.get(rel)
        if hit and hit[0] == fp and has_thumbs(cache, rel):
            vectors[rel] = hit[1]
        else:
            todo.append(rel)
    print(f"[piqa] {len(rels)} images in {root}  ({len(vectors)} cached, {len(todo)} to embed)")

    if todo:
        device = resolve_device()
        half = device.type == "cuda" and not fp32
        print(f"[piqa] loading {cfg.model} on {device}{' (fp16)' if half else ''}")
        model = load_model(cfg, device, half)

        workers = min(8, os.cpu_count() or 1) if num_workers is None else num_workers
        loader = DataLoader(
            _ImageDataset(root, todo, img_size),
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        t0 = time.time()
        with torch.inference_mode():
            for imgs, idx, ok in tqdm(loader, desc="embed", unit="batch"):
                imgs = imgs.to(device, non_blocking=True)
                feats = model(imgs.half() if half else imgs).float().cpu().numpy()
                for row, i, good in zip(feats, idx.tolist(), ok.tolist(), strict=True):
                    if good:
                        vectors[todo[i]] = row
        elapsed = time.time() - t0
        print(
            f"[piqa] embedded {len(todo)} images in {elapsed:.1f}s "
            f"({len(todo) / max(elapsed, 1e-9):.1f} img/s)"
        )

    keep = [i for i, rel in enumerate(rels) if rel in vectors]
    if not keep:
        raise SystemExit("No image could be decoded")
    np.savez(
        out,
        paths=np.array([rels[i] for i in keep]),
        fingerprints=np.array([fingerprints[i] for i in keep]),
        embeddings=np.stack([vectors[rels[i]] for i in keep]).astype(np.float32),
        model=np.array(cfg.model),
    )
    print(f"[piqa] saved {len(keep)} embeddings -> {out}")
    return out
