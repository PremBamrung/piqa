import json

import numpy as np
from PIL import Image

from piqa.grid import grid_shape, nearest_neighbours, snap_to_grid, spread, twin_links
from piqa.images import thumb_path
from piqa.viewers import Layout, build_atlas, build_payload, render


def blobs(n_per: int = 60, seed: int = 0) -> np.ndarray:
    """Four separated clusters, like UMAP output for four scenes."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
    return np.concatenate([c + rng.normal(size=(n_per, 2)) for c in centres])


def assert_one_photo_per_cell(cells, cols, rows):
    assert len({tuple(c) for c in cells.tolist()}) == len(cells)
    assert cells[:, 0].min() >= 0 and cells[:, 0].max() < cols
    assert cells[:, 1].min() >= 0 and cells[:, 1].max() < rows


def test_grid_shape_covers_every_photo():
    for n in (1, 2, 7, 100, 2554):
        for aspect in (0.5, 1.0, 2.0):
            cols, rows = grid_shape(n, aspect)
            assert cols * rows >= n


def test_grid_gives_every_photo_its_own_cell_and_keeps_clusters_together():
    pts = blobs()
    cells, cols, rows = snap_to_grid(pts)
    assert_one_photo_per_cell(cells, cols, rows)
    # Each cluster stays on its own side of the grid: its cells are closer to each other
    # than to the other clusters' cells.
    centroids = np.array([cells[i * 60 : (i + 1) * 60].mean(axis=0) for i in range(4)])
    for i in range(4):
        own = np.linalg.norm(cells[i * 60 : (i + 1) * 60] - centroids[i], axis=1).mean()
        others = min(np.linalg.norm(centroids[i] - centroids[j]) for j in range(4) if j != i)
        assert own < others


def test_big_layouts_are_split_into_tiles_with_the_same_guarantees():
    pts = blobs(n_per=100)
    cells, cols, rows = snap_to_grid(pts, max_tile=50)
    assert_one_photo_per_cell(cells, cols, rows)
    left = cells[pts[:, 0] < 5, 0].mean()
    right = cells[pts[:, 0] >= 5, 0].mean()
    assert left < right  # the split keeps the left clusters on the left


def test_spread_separates_piled_up_photos_and_keeps_clusters_apart():
    pts = blobs(n_per=60) * 0.05  # four tight piles, like bursts in UMAP
    pts[:5] = pts[0]  # five photos on exactly the same spot
    out = spread(pts, 0.8)
    d = np.linalg.norm(out[:, None] - out[None, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 0.75
    centroids = np.array([out[i * 60 : (i + 1) * 60].mean(axis=0) for i in range(4)])
    assert centroids[1, 0] > centroids[0, 0] and centroids[2, 1] > centroids[0, 1]


def test_spread_keeps_linked_twins_edge_to_edge():
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=0.05, size=(80, 2))  # one tight pile
    links = np.array([[i, i + 1] for i in range(0, 80, 2)])  # 40 twin pairs
    out = spread(pts, 1.1, links=links, near=0.75)
    d = np.linalg.norm(out[:, None] - out[None, :], axis=2)
    np.fill_diagonal(d, np.inf)
    twin_d = d[links[:, 0], links[:, 1]]
    assert d.min() >= 0.74  # nothing overlaps
    assert np.median(twin_d) < 0.85  # twins touch
    others = d.copy()
    others[links[:, 0], links[:, 1]] = others[links[:, 1], links[:, 0]] = np.inf
    assert others.min() >= 1.05  # everything else keeps a gap


def test_twin_links_pair_only_near_duplicates():
    x = np.array([[1, 0, 0], [0.99, 0.05, 0], [0, 1, 0], [0, 0.6, 0.8]], dtype=np.float32)
    assert twin_links(x, 0.9, k=2).tolist() == [[0, 1]]


def test_nearest_neighbours_are_ordered_and_never_self():
    x = np.array([[1, 0], [0.99, 0.1], [0.9, 0.4], [0, 1], [-1, 0]], dtype=np.float32)
    nb = nearest_neighbours(x, k=3, chunk=2)  # chunk < n exercises the chunking
    assert nb.shape == (5, 3)
    assert all(i not in row for i, row in enumerate(nb.tolist()))
    assert nb[0].tolist() == [1, 2, 3]


def test_atlas_packs_every_thumb_and_is_cached(tmp_path):
    cache = tmp_path / ".piqa"
    rels = [f"a/{i}.jpg" for i in range(5)]
    for i, rel in enumerate(rels):
        p = thumb_path(cache, rel, "s")
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 85) if i % 2 else (85, 128), "gray").save(p)
    entries = build_atlas(cache, rels, ["fp"] * 5)
    assert len(entries) == 5
    (_, _, _, w0, h0), (_, _, _, w1, h1) = entries[:2]
    assert w0 < h0 == 64 and h1 < w1 == 64  # portrait, landscape
    assert (cache / "atlas" / "64-0.jpg").exists()
    mtime = (cache / "atlas" / "64-0.jpg").stat().st_mtime_ns
    assert build_atlas(cache, rels, ["fp"] * 5) == entries
    assert (cache / "atlas" / "64-0.jpg").stat().st_mtime_ns == mtime  # not rebuilt


def test_payload_has_cells_groups_neighbours_and_free_positions():
    rels = ["Japon/Prem/été 1.JPG", "Japon/Laura/b.JPG", "Maroc/c.JPG", "Maroc/d.JPG"]
    coords = np.array([[0.0, 0.0], [2.0, 1.0], [10.0, 10.0], [11.0, 10.0]])
    layout = Layout(rels, np.eye(4), ["fp"] * 4, coords, "k")
    cells, cols, rows = snap_to_grid(coords)
    atlas = [[0, 0, 0, 64, 43]] * 4
    payload = build_payload(
        layout,
        cells,
        cols,
        rows,
        coords / 11,
        atlas,
        nearest_neighbours(np.eye(4), 2),
        color_by="top",
        initial_layout="grid",
    )
    first = payload["items"][0]
    assert first["p"] == "Japon/Prem/été 1.JPG"  # the page builds the encoded URLs
    assert [(g["name"], g["count"]) for g in payload["groups"]] == [("Japon", 2), ("Maroc", 2)]
    assert (first["c"], first["r"]) == tuple(cells[0].tolist())
    assert len(first["n"]) == 2 and 0 not in first["n"]
    assert 0 <= min(i["x"] for i in payload["items"]) and max(i["x"] for i in payload["items"]) <= 1
    assert payload["atlas"] == ["atlas/64-0.jpg"]


def test_render_escapes_the_title_and_embeds_the_data():
    page = render('<b>"trip"</b>', {"items": [{"p": "x</script>y"}]})
    assert "&lt;b&gt;&quot;trip&quot;&lt;/b&gt;" in page
    assert "<b>" not in page.split("<script>")[0].split("<style>")[0]
    data = page.split("const DATA = ")[1].split(";\n")[0]
    assert json.loads(data.replace("<\\/", "</"))["items"][0]["p"] == "x</script>y"
