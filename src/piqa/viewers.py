"""UMAP projection of cached embeddings and the two HTML viewers (scatter map + image wall).

The viewers are written into `.piqa/` and load thumbnails from `.piqa/thumbs/` by relative
path, so they stay small and open instantly; originals are linked as `../<path>`.
"""

from __future__ import annotations

import colorsys
import json
import time
from importlib.resources import files
from pathlib import Path
from urllib.parse import quote

import numpy as np

from piqa.images import cache_dir


def reduce(
    emb_path: Path, *, n_neighbors: int = 15, min_dist: float = 0.1, seed: int | None = 42
) -> tuple[list[str], np.ndarray]:
    """2-D UMAP of L2-normalised embeddings (euclidean on unit vectors == cosine)."""
    data = np.load(emb_path)
    rels = data["paths"].tolist()
    x = data["embeddings"]
    if len(rels) < 4:
        raise SystemExit(f"Need at least 4 images for a map, got {len(rels)}")
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    import umap  # slow import (numba), only when needed

    print(f"[piqa] UMAP on {len(rels)} images (n_neighbors={n_neighbors}, min_dist={min_dist})")
    t0 = time.time()
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, len(rels) - 1),
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        n_jobs=1 if seed is not None else -1,  # a fixed seed forces single-threaded UMAP
    )
    coords = reducer.fit_transform(x)
    print(f"[piqa] UMAP done in {time.time() - t0:.1f}s")
    np.savez(emb_path.parent / "coords.npz", paths=np.array(rels), coords=coords.astype(np.float32))
    return rels, coords


def palette(n: int) -> list[str]:
    colors = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 0.65, 0.95)
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colors


def group_of(rel: str, color_by: str) -> str:
    parts = rel.split("/")[:-1]
    if color_by == "none" or not parts:
        return "photos"
    return parts[0] if color_by == "top" else "/".join(parts)


def build_payload(rels: list[str], coords: np.ndarray, color_by: str) -> dict:
    # Fit into [0, 1] keeping the aspect ratio of the projection.
    mins = coords.min(axis=0)
    span = float((coords.max(axis=0) - mins).max()) or 1.0
    norm = (coords - mins) / span

    group_names = sorted({group_of(rel, color_by) for rel in rels})
    group_index = {name: i for i, name in enumerate(group_names)}
    items = []
    for rel, (x, y) in zip(rels, norm.tolist(), strict=True):
        items.append(
            {
                "x": round(x, 5),
                "y": round(y, 5),
                "g": group_index[group_of(rel, color_by)],
                "name": rel.rsplit("/", 1)[-1],
                "dir": rel.rsplit("/", 1)[0] if "/" in rel else "",
                "s": "thumbs/s/" + quote(f"{rel}.jpg"),
                "m": "thumbs/m/" + quote(f"{rel}.jpg"),
                "orig": "../" + quote(rel),
            }
        )
    groups = [
        {"name": n, "color": c} for n, c in zip(group_names, palette(len(group_names)), strict=True)
    ]
    return {"items": items, "groups": groups}


def render(template: str, title: str, payload: dict, plotly_js: str = "") -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = files("piqa").joinpath("templates", template).read_text(encoding="utf-8")
    return (
        html.replace("{{TITLE}}", title)
        .replace("/*{{PLOTLY_JS}}*/", plotly_js)
        .replace("/*{{DATA}}*/", data)
    )


def write_viewers(
    root: Path,
    rels: list[str],
    coords: np.ndarray,
    *,
    color_by: str = "folder",
    title: str | None = None,
) -> tuple[Path, Path]:
    from plotly.offline import get_plotlyjs

    cache = cache_dir(root)
    title = title or f"piqa · {root.resolve().name}"
    payload = build_payload(rels, coords, color_by)

    map_html = cache / "map.html"
    wall_html = cache / "wall.html"
    map_html.write_text(render("map.html", title, payload, get_plotlyjs()), encoding="utf-8")
    wall_html.write_text(render("wall.html", title, payload), encoding="utf-8")
    print(f"[piqa] map  -> {map_html}")
    print(f"[piqa] wall -> {wall_html}")
    return map_html, wall_html
