"""Snap a 2-D layout (UMAP) to a grid: every photo gets its own cell, neighbours stay close.

Each block of at most `max_tile` points is a linear assignment (points → cell centres, squared
distance), which is O(n³). Bigger layouts are split like a k-d tree first: the grid rectangle
is halved along its longer side and the points are split at the matching quantile, so each
half gets as many cells as it needs and every block stays small.
"""

from __future__ import annotations

import math

import numpy as np


def grid_shape(n: int, aspect: float, cell_h: float = 1.0) -> tuple[int, int]:
    """Columns and rows (cols * rows >= n) whose overall shape is close to `aspect` (w / h)."""
    # width = cols, height = rows * cell_h and cols ≈ n / rows  →  rows = √(n / (aspect · cell_h))
    rows = max(1, round(math.sqrt(n / (aspect * cell_h))))
    rows = min(rows, n)
    return math.ceil(n / rows), rows


def snap_to_grid(
    coords: np.ndarray, *, cell_h: float = 1.0, max_tile: int = 3000
) -> tuple[np.ndarray, int, int]:
    """Cell (col, row) for every point, plus the grid's columns and rows.

    `cell_h` is a cell's height for a width of 1 (0.7 suits 3:2 photos), so distances are
    measured in the same units as the drawn wall.
    """
    from scipy.optimize import linear_sum_assignment

    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int32), 0, 0
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    pts = np.clip(pts, lo, hi)
    span = np.maximum(hi - lo, 1e-9)
    aspect = float(np.clip(span[0] / span[1], 1 / 3, 3))
    cols, rows = grid_shape(n, aspect, cell_h)
    cells = np.empty((n, 2), dtype=np.int32)

    def assign(idx: np.ndarray, c0: int, r0: int, w: int, h: int) -> None:
        p = pts[idx]
        p_lo = p.min(axis=0)
        p_span = np.maximum(p.max(axis=0) - p_lo, 1e-9)
        q = (p - p_lo) / p_span * np.array([w, h * cell_h])
        cx, cy = np.meshgrid(np.arange(w) + 0.5, (np.arange(h) + 0.5) * cell_h)
        cx, cy = cx.ravel(), cy.ravel()
        cost = (q[:, None, 0] - cx[None, :]) ** 2 + (q[:, None, 1] - cy[None, :]) ** 2
        point, cell = linear_sum_assignment(cost)
        cells[idx[point], 0] = c0 + cell % w
        cells[idx[point], 1] = r0 + cell // w

    def place(idx: np.ndarray, c0: int, r0: int, w: int, h: int) -> None:
        n_here = len(idx)
        if n_here <= max_tile or w * h <= 1:
            assign(idx, c0, r0, w, h)
            return
        split_cols = w * cell_h >= h  # halve the longer side (in drawn units)
        if split_cols:
            w1 = w // 2
            cap1, cap2 = w1 * h, (w - w1) * h
        else:
            h1 = h // 2
            cap1, cap2 = w * h1, w * (h - h1)
        n1 = min(max(round(n_here * cap1 / (cap1 + cap2)), n_here - cap2), cap1)
        order = idx[np.argsort(pts[idx, 0 if split_cols else 1], kind="stable")]
        if split_cols:
            place(order[:n1], c0, r0, w1, h)
            place(order[n1:], c0 + w1, r0, w - w1, h)
        else:
            place(order[:n1], c0, r0, w, h1)
            place(order[n1:], c0, r0 + h1, w, h - h1)

    place(np.arange(n), 0, 0, cols, rows)
    return cells, cols, rows


def spread(
    points: np.ndarray,
    min_dist: float,
    *,
    links: np.ndarray | None = None,
    near: float | None = None,
    pull_rounds: int = 1000,
    iterations: int = 3000,
    seed: int = 0,
) -> np.ndarray:
    """Push points apart until no two centres are closer than `min_dist`.

    UMAP packs similar photos (bursts, near-duplicates) onto almost the same spot. Each round
    finds the pairs that are too close and pushes them apart (to 1.2x the distance they need,
    so pairs don't hover at the limit), at most two spacings per round so crowds expand
    without overshooting.

    Pushing alone spreads a pile evenly and scatters its twins, so `links` (index pairs, i < j)
    may only be `near` apart and are pulled back to that distance for the first `pull_rounds`
    rounds; the remaining rounds only push, to clear what the pull squeezed. Twins end up
    edge to edge, in clumps with gaps between them.
    """
    from scipy.spatial import cKDTree

    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    rng = np.random.default_rng(seed)
    links = np.zeros((0, 2), dtype=np.int64) if links is None else np.asarray(links, np.int64)
    near = min_dist if near is None else near
    link_keys = np.sort(links[:, 0] * n + links[:, 1])
    for it in range(iterations):
        move = np.zeros_like(pts)
        settled = True
        pairs = cKDTree(pts).query_pairs(min_dist * 0.999, output_type="ndarray")
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            need = np.where(np.isin(a * n + b, link_keys), near, min_dist)
            delta = pts[b] - pts[a]
            dist = np.linalg.norm(delta, axis=1)
            close = dist < need * 0.999
            if close.any():
                settled = False
                a, b, need, delta, dist = a[close], b[close], need[close], delta[close], dist[close]
                same = dist < 1e-9  # identical spots: pick a random direction
                delta[same] = rng.normal(size=(int(same.sum()), 2))
                dist[same] = np.linalg.norm(delta[same], axis=1)
                gap = need * 1.2 - np.where(same, 0, dist)
                push = delta / dist[:, None] * (gap / 2)[:, None]
                np.add.at(move, a, -push)
                np.add.at(move, b, push)
        if len(links) and it < pull_rounds:
            a, b = links[:, 0], links[:, 1]
            delta = pts[b] - pts[a]
            dist = np.linalg.norm(delta, axis=1)
            apart = dist > near * 1.05
            if apart.any():
                settled = False
                unit = delta[apart] / dist[apart, None]
                pull = unit * ((dist[apart] - near) / 4)[:, None]
                np.add.at(move, a[apart], pull)
                np.add.at(move, b[apart], -pull)
        if settled:
            break
        norm = np.linalg.norm(move, axis=1, keepdims=True)
        pts += move * np.minimum(1, 2 * near / np.maximum(norm, 1e-12))
    return pts


def twin_links(vectors: np.ndarray, threshold: float = 0.9, k: int = 3) -> np.ndarray:
    """Index pairs (i < j) of near-duplicates: j is among i's `k` most similar, cosine >= threshold.

    0.9 is where consecutive burst frames sit (median 0.96, RECOMMENDATIONS §1); `k` keeps a
    long burst a chain of links rather than a clique that can't all sit edge to edge.
    """
    x = np.asarray(vectors, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    nb = nearest_neighbours(x, k)
    if not nb.size:
        return np.zeros((0, 2), dtype=np.int64)
    i = np.repeat(np.arange(len(x)), nb.shape[1])
    j = nb.ravel()
    keep = np.einsum("ij,ij->i", x[i], x[j]) >= threshold
    pairs = np.sort(np.stack([i[keep], j[keep]], axis=1), axis=1)
    return np.unique(pairs, axis=0).astype(np.int64)


def nearest_neighbours(vectors: np.ndarray, k: int = 12, chunk: int = 1024) -> np.ndarray:
    """Indices of the `k` most cosine-similar vectors for each row, most similar first."""
    x = np.asarray(vectors, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n = len(x)
    k = min(k, n - 1)
    out = np.empty((n, max(k, 0)), dtype=np.int32)
    if k <= 0:
        return out
    for start in range(0, n, chunk):
        sims = x[start : start + chunk] @ x.T
        rows = np.arange(sims.shape[0])
        sims[rows, start + rows] = -np.inf  # never your own neighbour
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
        out[start : start + chunk] = np.take_along_axis(top, order, axis=1)
    return out
