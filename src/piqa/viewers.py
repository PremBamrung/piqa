"""The wall: UMAP layout of cached embeddings, snapped to a grid, drawn from a thumbnail atlas.

Everything is written into `.piqa/` and loaded by relative path, so the page stays small and
opens instantly; originals are linked as `../<path>`. Each step is cached and only redone
when its inputs change:

- `coords.npz`: UMAP coordinates (embeddings + UMAP settings)
- `grid.npz`: grid cells for those coordinates, and the free layout spread so photos don't
  sit on top of each other
- `atlas/`: every photo at 64 px packed into a few sprite sheets, so the overview draws from
  2-3 images instead of thousands
"""

from __future__ import annotations

import colorsys
import hashlib
import html
import json
import time
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np

from piqa.grid import nearest_neighbours, snap_to_grid, spread, twin_links
from piqa.images import cache_dir, thumb_path

CELL_H = 1.0  # square cells: landscape and portrait photos get the same space
ATLAS_TILE = 64
ATLAS_SHEET = 2048
NEIGHBOURS = 12
# Free layout, in grid cells: a light spread, so UMAP's shape and empty space survive (photos
# move ~1-2% of the layout's width). Photo centres at least FREE_GAP apart, near-duplicates
# (cosine >= TWIN_COSINE) only FREE_TWIN apart, i.e. edge to edge for the page's 0.4-wide
# tiles. Piles open up as you zoom and are fully separate at ~9x the fitted zoom.
FREE_GAP = 0.5
FREE_TWIN = 0.42
TWIN_COSINE = 0.9


@dataclass
class Layout:
    rels: list[str]
    vectors: np.ndarray
    fingerprints: list[str]
    coords: np.ndarray
    key: str  # changes whenever the coordinates do


def _digest(*parts: object) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.tobytes() if isinstance(part, np.ndarray) else repr(part).encode())
    return h.hexdigest()[:16]


def _cached(path: Path, key: str) -> dict | None:
    if not path.exists():
        return None
    try:
        data = np.load(path)
        return dict(data) if str(data["key"]) == key else None
    except (OSError, KeyError, ValueError):
        return None


def reduce(
    emb_path: Path, *, n_neighbors: int = 15, min_dist: float = 0.1, seed: int | None = 42
) -> Layout:
    """2-D UMAP of L2-normalised embeddings (euclidean on unit vectors == cosine), cached."""
    data = np.load(emb_path)
    rels = data["paths"].tolist()
    x = data["embeddings"]
    fingerprints = data["fingerprints"].tolist()
    if len(rels) < 4:
        raise SystemExit(f"Need at least 4 images for a wall, got {len(rels)}")
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    key = _digest(rels, x, n_neighbors, min_dist, seed)
    out = emb_path.parent / "coords.npz"
    if (hit := _cached(out, key)) is not None:
        print("[piqa] UMAP: cached")
        return Layout(rels, x, fingerprints, hit["coords"], key)

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
    coords = reducer.fit_transform(x).astype(np.float32)
    print(f"[piqa] UMAP done in {time.time() - t0:.1f}s")
    np.savez(out, paths=np.array(rels), coords=coords, key=np.array(key))
    return Layout(rels, x, fingerprints, coords, key)


def place(cache: Path, layout: Layout) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Grid cells (col, row), grid size, and free-layout positions in [0, ~1] (x and y / cols)."""
    key = _digest(layout.key, CELL_H, FREE_GAP, FREE_TWIN, TWIN_COSINE)
    out = cache / "grid.npz"
    if (hit := _cached(out, key)) is not None:
        return hit["cells"], int(hit["cols"]), int(hit["rows"]), hit["free"]
    t0 = time.time()
    cells, cols, rows = snap_to_grid(layout.coords, cell_h=CELL_H)
    print(f"[piqa] grid {cols}x{rows} in {time.time() - t0:.1f}s")
    # Free layout in grid units (the page draws it `cols` wide), spread so that photos get
    # their own place once zoomed in, with near-duplicates kept edge to edge.
    t0 = time.time()
    links = twin_links(layout.vectors, TWIN_COSINE)
    world = spread(free_positions(layout.coords) * cols, FREE_GAP, links=links, near=FREE_TWIN)
    free = ((world - world.min(axis=0)) / cols).astype(np.float32)
    print(f"[piqa] free layout spread in {time.time() - t0:.1f}s")
    np.savez(out, cells=cells, cols=cols, rows=rows, free=free, key=np.array(key))
    return cells, cols, rows, free


def build_atlas(cache: Path, rels: list[str], fingerprints: list[str]) -> list[list[int]]:
    """Pack every small thumbnail at ATLAS_TILE px into sheets; [sheet, x, y, w, h] per photo."""
    from PIL import Image

    folder = cache / "atlas"
    index = folder / "index.json"
    key = _digest(rels, fingerprints, ATLAS_TILE, ATLAS_SHEET)
    try:
        saved = json.loads(index.read_text(encoding="utf-8"))
        if saved["key"] == key:
            return saved["entries"]
    except (OSError, ValueError, KeyError):
        pass

    t0 = time.time()
    folder.mkdir(exist_ok=True)
    per_row = ATLAS_SHEET // ATLAS_TILE
    per_sheet = per_row * per_row
    entries = []
    sheet = None
    for i, rel in enumerate(rels):
        n, slot = divmod(i, per_sheet)
        if slot == 0:
            if sheet is not None:
                sheet.save(folder / f"{ATLAS_TILE}-{n - 1}.jpg", quality=85)
            remaining = min(per_sheet, len(rels) - i)
            height = -(-remaining // per_row) * ATLAS_TILE
            sheet = Image.new("RGB", (ATLAS_SHEET, height), (10, 10, 10))
        x0, y0 = (slot % per_row) * ATLAS_TILE, (slot // per_row) * ATLAS_TILE
        try:
            with Image.open(thumb_path(cache, rel, "s")) as im:
                im = im.convert("RGB")
                im.thumbnail((ATLAS_TILE, ATLAS_TILE))
                x, y = x0 + (ATLAS_TILE - im.width) // 2, y0 + (ATLAS_TILE - im.height) // 2
                sheet.paste(im, (x, y))
                entries.append([n, x, y, im.width, im.height])
        except OSError:
            entries.append([n, x0, y0, ATLAS_TILE, ATLAS_TILE * 2 // 3])
    if sheet is not None:
        sheet.save(folder / f"{ATLAS_TILE}-{(len(rels) - 1) // per_sheet}.jpg", quality=85)
    for stale in folder.glob(f"{ATLAS_TILE}-*.jpg"):
        if int(stale.stem.split("-")[1]) > (len(rels) - 1) // per_sheet:
            stale.unlink()
    index.write_text(json.dumps({"key": key, "entries": entries}), encoding="utf-8")
    print(f"[piqa] atlas: {len(rels)} tiles in {time.time() - t0:.1f}s")
    return entries


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


def free_positions(coords: np.ndarray) -> np.ndarray:
    """UMAP coordinates fitted into [0, 1], aspect kept, outliers clipped to the 1st/99th pct."""
    lo, hi = np.percentile(coords, 1, axis=0), np.percentile(coords, 99, axis=0)
    span = float((hi - lo).max()) or 1.0
    return (np.clip(coords, lo, hi) - lo) / span


def build_payload(
    layout: Layout,
    cells: np.ndarray,
    cols: int,
    rows: int,
    free: np.ndarray,
    atlas: list[list[int]],
    neighbours: np.ndarray,
    *,
    color_by: str,
    initial_layout: str,
) -> dict:
    group_names = sorted({group_of(rel, color_by) for rel in layout.rels})
    group_index = {name: i for i, name in enumerate(group_names)}
    gs = [group_index[group_of(rel, color_by)] for rel in layout.rels]
    items = [
        {
            "p": rel,
            "g": g,
            "c": int(c),
            "r": int(r),
            "x": round(float(x), 5),
            "y": round(float(y), 5),
            "a": a,
            "n": nb,
        }
        for rel, g, (c, r), (x, y), a, nb in zip(
            layout.rels, gs, cells.tolist(), free.tolist(), atlas, neighbours.tolist(), strict=True
        )
    ]
    groups = [
        {"name": name, "color": color, "count": gs.count(i)}
        for i, (name, color) in enumerate(zip(group_names, palette(len(group_names)), strict=True))
    ]
    sheets = (max(a[0] for a in atlas) + 1) if atlas else 0
    return {
        "cols": cols,
        "rows": rows,
        "cellH": CELL_H,
        "layout": initial_layout,
        "atlas": [f"atlas/{ATLAS_TILE}-{n}.jpg" for n in range(sheets)],
        "groups": groups,
        "items": items,
    }


def render(title: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = files("piqa").joinpath("templates", "wall.html").read_text(encoding="utf-8")
    return page.replace("{{TITLE}}", html.escape(title)).replace("/*{{DATA}}*/", data)


def write_wall(
    root: Path,
    layout: Layout,
    *,
    color_by: str = "folder",
    initial_layout: str = "grid",
    title: str | None = None,
) -> Path:
    cache = cache_dir(root)
    title = title or f"piqa · {root.resolve().name}"
    cells, cols, rows, free = place(cache, layout)
    atlas = build_atlas(cache, layout.rels, layout.fingerprints)
    neighbours = nearest_neighbours(layout.vectors, NEIGHBOURS)
    payload = build_payload(
        layout,
        cells,
        cols,
        rows,
        free,
        atlas,
        neighbours,
        color_by=color_by,
        initial_layout=initial_layout,
    )
    wall_html = cache / "wall.html"
    wall_html.write_text(render(title, payload), encoding="utf-8")
    (cache / "map.html").unlink(missing_ok=True)  # the old scatter viewer, now gone
    print(f"[piqa] wall -> {wall_html}")
    return wall_html
