# Unified Image Mapping Pipeline — Implementation Plan

> **Historical.** Written in May 2026 for the standalone scripts that piqa replaced. Script names, flags and line numbers refer to those scripts. What piqa adopted: [CONTEXT.md §5](CONTEXT.md#5-earlier-plans-what-piqa-adopted).

> Goal: one coherent pipeline that keeps Pipeline 2's semantic quality (DINOv3, modular stages, canvas wall) and brings in everything good from Pipeline 1 (WebGL scatter, persistent sidebar, offline output, subsample flag, MPS support, arg validation).

---

## Target Architecture

```
photo-map run \
  --images_dir Source/Laura/Atacama \
  --model      dinov3_vits16plus \
  --weights    dinov3/dinov3_vits16plus.pth \
  --repo_dir   dinov3_repo \
  [--max_images 500]           # optional subset for fast iteration
  [--n_neighbors 15]
  [--min_dist 0.1]
  [--colorby subfolder]
  [--offline]                  # bundle Plotly inline
  [--output_dir out/]

Stages executed in order:
  1. extract_embeddings.py  →  out/embeddings.npz
  2. umap_reduce.py         →  out/umap_coords.npz
  3. plot_viewer.py         →  out/photo_map.html    (scatter, sidebar, WebGL)
  4. wall_viewer.py         →  out/photo_wall.html   (canvas wall)
```

All four HTML outputs are fully self-contained and offline.

---

## Files to Create / Modify

| File | Action | Summary |
|---|---|---|
| `photo_map.py` | **Create** | New unified entry-point that orchestrates all 4 stages end-to-end |
| `extract_embeddings.py` | **Modify** | Add `--max_images`, MPS support, `validate_args()` |
| `umap_reduce.py` | **Modify** | Minor: surface `--seed` consistently, add `validate_args()` |
| `plot_viewer.py` | **Modify** | Switch to `ScatterGL`, add persistent sidebar, add `--offline` flag |
| `wall_viewer.py` | No change | Already good as-is |

---

## Step-by-Step Implementation

---

### Step 1 — Add `--max_images` and MPS to `extract_embeddings.py`

**Where:** `extract_embeddings.py`, `main()` function and `ImageFolder.__init__()`

**Changes:**

```python
# 1. Add to argparse
parser.add_argument(
    "--max_images",
    type=int,
    default=None,
    help="Cap number of images (useful for fast prototyping)",
)

# 2. Truncate after sorting in ImageFolder.__init__
if max_images is not None:
    self.paths = self.paths[:max_images]


# 3. Expand device resolution to include MPS
def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

**Also add `validate_args()`:**
```python
def validate_args(args):
    if not args.images_dir.exists():
        raise SystemExit(f"--images_dir not found: {args.images_dir}")
    if not args.repo_dir.exists():
        raise SystemExit(f"--repo_dir not found: {args.repo_dir}")
    if not Path(args.weights).exists():
        raise SystemExit(f"--weights not found: {args.weights}")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive")
```

**Note on fp16 + MPS:** MPS does not support float16 reliably — add a guard:
```python
if args.fp16 and device.type == "mps":
    print("[WARN]  fp16 not supported on MPS — ignoring --fp16")
    args.fp16 = False
```

---

### Step 2 — Switch `plot_viewer.py` to `ScatterGL` + persistent sidebar + offline mode

This is the biggest change. Three independent improvements bundled into one file.

#### 2a. `ScatterGL` (WebGL rendering)

In `plot_viewer.py`, change `"type": "scatter"` to `"type": "scattergl"` in the trace dict.

```python
# Before (line ~157)
trace = {
    "type": "scatter",
    ...
}

# After
trace = {
    "type": "scattergl",
    ...
}
```

That's it. Everything else (marker, customdata, hovertemplate) stays identical.

#### 2b. Persistent sidebar

Replace the current full-viewport layout with a two-column grid:

```
┌─────────────────────────────┬──────────────────┐
│                             │  [ Image Preview ]│
│      Plotly scatter         │                   │
│      (left column,          │  filename.jpg     │
│       ~75% width)           │  x=1.234 y=−0.567 │
│                             │                   │
│                             │  (persists after  │
│                             │   mouse leaves)   │
└─────────────────────────────┴──────────────────┘
```

**CSS change:**
```css
body {
  display: grid;
  grid-template-columns: 1fr 280px;
  height: 100vh;
  background: #0f0f0f;
  overflow: hidden;
}
#sidebar {
  background: #141414;
  border-left: 1px solid #222;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow-y: auto;
}
#sidebar img {
  width: 100%;
  border-radius: 4px;
  object-fit: contain;
  background: #1a1a1a;
}
#sidebar-name  { font-size: 12px; color: #888; font-family: monospace; word-break: break-all; }
#sidebar-coords { font-size: 11px; color: #555; font-family: monospace; }
#sidebar-empty  { color: #444; font-size: 13px; }
```

**JS change — replace floating overlay logic with sidebar update:**
```javascript
const sidebar     = document.getElementById('sidebar');
const sidebarImg  = document.getElementById('sidebar-img');
const sidebarName = document.getElementById('sidebar-name');
const sidebarCoords = document.getElementById('sidebar-coords');

gd.on('plotly_hover', function(ev) {
  const pt    = ev.points[0];
  const thumb = pt.customdata;
  if (!thumb) return;
  sidebarImg.src          = thumb;
  sidebarImg.style.display = 'block';
  sidebarName.textContent  = pt.text.split('<br>')[0];
  sidebarCoords.textContent = pt.text.split('<br>')[1] || '';
  document.getElementById('sidebar-empty').style.display = 'none';
});
// No plotly_unhover handler — sidebar persists last image intentionally
```

Remove the floating `div#img-overlay` entirely.

#### 2c. `--offline` flag (inline Plotly)

```python
parser.add_argument(
    "--offline", action="store_true", help="Embed Plotly JS inline (no CDN dependency)"
)
```

In HTML generation:
```python
if args.offline:
    import urllib.request

    plotly_js_url = "https://cdn.plot.ly/plotly-2.35.2.min.js"
    with urllib.request.urlopen(plotly_js_url) as r:
        plotly_js = r.read().decode("utf-8")
    plotly_script = f"<script>{plotly_js}</script>"
else:
    plotly_script = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
```

Replace the hardcoded `<script src="...">` tag with `{plotly_script}` in the HTML template.

---

### Step 3 — Create `photo_map.py` (unified entry-point)

New file. Orchestrates all four stages by calling each script's `main()` after injecting the right `sys.argv`, or better — import and call each module's `main()` directly.

**Preferred approach:** each stage exposes a `run(args)` function; `photo_map.py` calls them in order. This avoids subprocess overhead and keeps all output in one terminal session.

**Refactor required in each stage script:**

```python
# In each stage script, wrap existing main() body:
def run(args: argparse.Namespace) -> None:
    # ... existing main() body ...

def main() -> None:
    args = parse_args()
    validate_args(args)
    run(args)

if __name__ == "__main__":
    main()
```

**`photo_map.py` structure:**

```python
#!/usr/bin/env python3
"""
Unified photo mapping pipeline.
Runs all 4 stages: extract → reduce → scatter viewer → wall viewer.
"""

import argparse
from pathlib import Path
import extract_embeddings
import umap_reduce
import plot_viewer
import wall_viewer


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end photo semantic map")

    # Stage 1 — embedding
    p.add_argument("--images_dir", required=True, type=Path)
    p.add_argument("--repo_dir", required=True, type=Path)
    p.add_argument("--weights", required=True, type=str)
    p.add_argument("--model", default="dinov3_vits16plus")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Process only the first N images (fast prototyping)",
    )

    # Stage 2 — UMAP
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--metric", default="cosine", choices=["cosine", "euclidean", "correlation"])
    p.add_argument("--seed", type=int, default=42)

    # Stage 3 — scatter viewer
    p.add_argument("--thumb_size", type=int, default=200)
    p.add_argument("--point_size", type=int, default=6)
    p.add_argument("--colorby", default="none", choices=["none", "subfolder", "index"])
    p.add_argument(
        "--offline", action="store_true", help="Embed Plotly inline (no internet needed)"
    )

    # Stage 4 — wall viewer
    p.add_argument("--wall_thumb", type=int, default=96, help="Thumbnail size for canvas wall")

    # Output
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("."),
        help="Directory for all output files (default: current dir)",
    )

    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    emb_path = args.output_dir / "embeddings.npz"
    coords_path = args.output_dir / "umap_coords.npz"
    map_path = args.output_dir / "photo_map.html"
    wall_path = args.output_dir / "photo_wall.html"

    print("\n═══ STAGE 1 / 4 — Feature Extraction ═══")
    extract_embeddings.run(
        argparse.Namespace(
            images_dir=args.images_dir,
            repo_dir=args.repo_dir,
            weights=args.weights,
            model=args.model,
            output=emb_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            fp16=args.fp16,
            max_images=args.max_images,
        )
    )

    print("\n═══ STAGE 2 / 4 — UMAP Reduction ═══")
    umap_reduce.run(
        argparse.Namespace(
            input=emb_path,
            output=coords_path,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            n_components=2,
            random_state=args.seed,
            low_memory=False,
        )
    )

    print("\n═══ STAGE 3 / 4 — Scatter Viewer ═══")
    plot_viewer.run(
        argparse.Namespace(
            coords=coords_path,
            images_dir=args.images_dir,
            output=map_path,
            thumb_size=args.thumb_size,
            point_size=args.point_size,
            colorby=args.colorby,
            offline=args.offline,
            title="📷 Photo Map — DINOv3 + UMAP",
        )
    )

    print("\n═══ STAGE 4 / 4 — Canvas Wall ═══")
    wall_viewer.run(
        argparse.Namespace(
            coords=coords_path,
            images_dir=args.images_dir,
            output=wall_path,
            thumb_size=args.wall_thumb,
            num_workers=args.num_workers,
        )
    )

    print(f"""
╔══════════════════════════════════════════════╗
║  Done.  Open either viewer:                  ║
║    file://{map_path.resolve()}
║    file://{wall_path.resolve()}
╚══════════════════════════════════════════════╝""")


if __name__ == "__main__":
    main()
```

---

### Step 4 — Minor: `umap_reduce.py` hardening

Add `validate_args()` and expose `run(args)`:

```python
def validate_args(args):
    if not args.input.exists():
        raise SystemExit(f"--input not found: {args.input}")
    if args.n_neighbors < 2:
        raise SystemExit("--n_neighbors must be >= 2")
    if not (0.0 <= args.min_dist <= 1.0):
        raise SystemExit("--min_dist must be in [0.0, 1.0]")
```

---

## Implementation Order

Work in this sequence — each step is independently testable:

```
1. extract_embeddings.py  → add max_images + MPS + validate_args + run()
2. umap_reduce.py         → add validate_args + run()
3. plot_viewer.py         → ScatterGL + sidebar + offline flag + run()
4. photo_map.py           → create orchestrator (depends on steps 1–3)
5. Smoke-test             → run photo_map.py --max_images 50 on a small folder
```

---

## Smoke Test

```bash
python photo_map.py \
  --images_dir "Source/Laura/Atacama Uyuni/Prem/2026-03-31/JPG" \
  --repo_dir   dinov3_repo \
  --weights    "dinov3/dinov3_vits16plus.pth" \
  --max_images 50 \
  --n_neighbors 10 \
  --min_dist 0.05 \
  --colorby subfolder \
  --offline \
  --output_dir out/test
```

Expected: both `out/test/photo_map.html` and `out/test/photo_wall.html` open in the browser, 50 images visible, scatter uses WebGL, sidebar persists on unhover, no CDN requests.

---

## What Does NOT Change

- `wall_viewer.py` — already well-implemented; no changes needed.
- DINOv3 model loading logic — works, leave it alone.
- UMAP hyperparameter set — already well-exposed.
- The `.npz` intermediate artifact format — keeps backward compatibility with existing cached embeddings.
- The four scripts remain independently runnable — `photo_map.py` is an addition, not a replacement.

---

## Summary of All Changes

| Change | File | Lines | Value |
|---|---|---|---|
| `scattergl` instead of `scatter` | `plot_viewer.py` | 1 | WebGL rendering — fluent at 5k+ points |
| Persistent sidebar | `plot_viewer.py` | ~50 | Study images without holding cursor still |
| `--offline` Plotly bundling | `plot_viewer.py` | ~15 | Truly portable output |
| `--max_images` cap | `extract_embeddings.py` | ~10 | Fast prototyping subset |
| MPS device support | `extract_embeddings.py` | ~10 | Apple Silicon compatibility |
| `validate_args()` | both stage scripts | ~20 | Clear errors instead of tracebacks |
| `run(args)` refactor | all 4 stage scripts | ~5 each | Enables orchestration |
| `photo_map.py` orchestrator | new file | ~90 | Single-command pipeline |
