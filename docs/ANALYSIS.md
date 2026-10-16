# Image Mapping Pipelines — Full Analysis

> **Historical.** Written in May 2026 for the standalone scripts that piqa replaced. Script names, flags and line numbers refer to those scripts. What piqa adopted: [CONTEXT.md §5](CONTEXT.md#5-earlier-plans-what-piqa-adopted).

> Working directory: `/mnt/g/Users/Prem/Pictures/Photography`
> Date: 2026-05-02

---

## Overview

Two independent pipelines exist for turning a photo library into an interactive semantic map. Both embed images into a high-dimensional vector space and project to 2D via UMAP, but they differ in architecture, model quality, output capability, and reusability.

```
PIPELINE 1 (Single-Stage):
[JPEGs] ──► image_map.py ──► [ResNet50 2048-d] ──► [UMAP 2D] ──► [Plotly HTML]

PIPELINE 2 (Multi-Stage):
[Images] ──► extract_embeddings.py ──► [DINOv3 384–1024-d] ──► embeddings.npz
                                                                       │
                                                              umap_reduce.py
                                                                       │
                                                             umap_coords.npz
                                                            ┌──────────┴──────────┐
                                                   plot_viewer.py        wall_viewer.py
                                                        │                       │
                                                photo_map.html          photo_wall.html
```

---

## Pipeline 1 — `image_map.py` (ResNet50, single script, 422 lines)

### What it does

One command reads all JPEGs recursively → loads pretrained ResNet50 (ImageNet) → strips the classification head → extracts 2048-d feature vectors → runs UMAP → generates a Plotly scatter with hover thumbnails → writes a single self-contained HTML file.

### Strengths

| # | Strength | Detail |
|---|---|---|
| 1 | **Zero friction** | One command, one output file. No staging, no intermediate artifacts. |
| 2 | **CPU / MPS friendly** | Explicit `resolve_device()` supports `auto / cpu / mps` — works well on Apple Silicon without a GPU. |
| 3 | **Truly offline output** | Uses `figure.to_html(include_plotlyjs=True)` — Plotly is bundled inline. No internet required, ever. |
| 4 | **WebGL scatter (`ScatterGL`)** | Uses `go.Scattergl` — GPU-accelerated rendering in the browser. Stays fluid at thousands of points. |
| 5 | **Persistent sidebar** | `<aside>` panel holds the last-hovered image after the mouse stops. You can study it without keeping the cursor still. |
| 6 | **Square thumbnail canvas** | `image_to_data_uri()` centres the thumbnail on a fixed square canvas — consistent visual grid in the sidebar. |
| 7 | **`--max-images` cap** | Optional hard limit for quick prototyping without processing the entire library. |
| 8 | **Arg validation** | `validate_args()` catches bad CLI inputs early with clear error messages. |
| 9 | **Reproducibility** | Seeds `random`, `numpy`, and `torch` together via `set_seed()`. |

### Weak Points

| # | Weakness | Impact |
|---|---|---|
| 1 | **Shallow visual semantics** | ResNet50 was trained to classify 1000 ImageNet objects. It groups "outdoor person" together regardless of lighting, mood, colour palette, or composition — the things that matter for photography. |
| 2 | **No reusable artifacts** | Re-running with different UMAP params means a full re-embed. No way to experiment with the projection without paying the full inference cost each time. |
| 3 | **JPEG-only** | Hard-coded `{".jpg", ".jpeg"}` — ignores PNG, TIFF, WebP, BMP. |
| 4 | **Single visualisation** | Scatter plot only — no spatial canvas wall view. |
| 5 | **Non-modular** | 420-line monolith. Adding a new view or coloring scheme requires editing a single large file. |
| 6 | **No GPU path (CUDA)** | MPS only — no CUDA support for Linux / WSL setups. |

---

## Pipeline 2 — DINOv3 Multi-Stage (4 scripts)

### What it does

Four decoupled stages. Stage 1 embeds images with a DINOv3 Vision Transformer (trained on 1.6 billion images) and saves `embeddings.npz`. Stage 2 reduces to 2D and saves `umap_coords.npz`. Stages 3 and 4 consume those coordinates to produce two HTML viewers independently.

### Stage breakdown

| Script | Input | Output | Key operation |
|---|---|---|---|
| `extract_embeddings.py` | image dir | `embeddings.npz` | DINOv3 [CLS] token extraction |
| `umap_reduce.py` | `embeddings.npz` | `umap_coords.npz` | UMAP dimensionality reduction |
| `plot_viewer.py` | `umap_coords.npz` | `photo_map.html` | Plotly scatter with hover overlay |
| `wall_viewer.py` | `umap_coords.npz` | `photo_wall.html` | Zoomable HTML5 Canvas wall |

### Strengths

| # | Strength | Detail |
|---|---|---|
| 1 | **Far superior semantics** | DINOv3 (ViT, self-supervised on LVD-1689M) understands scene composition, colour tone, depth of field, bokeh, and subject similarity — not just object class labels. The resulting map is genuinely useful for a photographer. |
| 2 | **Artifact reuse** | `embeddings.npz` is computed once. UMAP re-runs are nearly instant — freely experiment with `n_neighbors`, `min_dist`, `metric` without touching the model. |
| 3 | **Two complementary views** | `photo_map.html` for cluster exploration; `photo_wall.html` for a gestalt overview with every image visible at once — spatial arrangement communicates the library structure immediately. |
| 4 | **GPU-ready with float16** | `--fp16` flag, `pin_memory`, `prefetch_factor=2`, multi-worker DataLoader — scales to large libraries fast on CUDA hardware. |
| 5 | **Multi-format input** | JPEG, PNG, TIFF, WebP, BMP — 7 extensions covered. |
| 6 | **Aspect-ratio-preserving preprocessing** | `PadToSquare` pads to square with black before resizing — prevents aspect ratio from leaking into the embedding space. |
| 7 | **Colouring modes** | Scatter can be coloured by subfolder (shoot/trip), index, or flat — useful for seeing how date or location correlates with semantic clusters. |
| 8 | **Canvas wall UX** | Zoom, pan, pinch (touch), click-to-lightbox, filename tooltip, reset button — all in ~250 lines of vanilla JS. |
| 9 | **Modular / composable** | Each stage is a standalone script — you can swap in a different model or a different visualiser without touching the others. |

### Weak Points

| # | Weakness | Impact |
|---|---|---|
| 1 | **Setup complexity** | Requires `dinov3_repo` clone + `.pth` checkpoint files (100–500 MB). Not plug-and-play. |
| 2 | **4 commands, no orchestration** | No glue tying the stages together. New users must remember the correct sequence and flags. |
| 3 | **SVG scatter (not WebGL)** | `plot_viewer.py` emits `"type": "scatter"` (line 157) — SVG-based. Noticeably slow above ~2000 points in the browser. |
| 4 | **CDN-dependent output** | `plot_viewer.py` loads Plotly from `cdn.plot.ly` — not truly offline until first load has been cached. |
| 5 | **No max-images / subsample flag** | No way to quickly run the pipeline on a random subset for prototyping or parameter tuning. |
| 6 | **Disappearing hover overlay** | The floating image div in `plot_viewer.py` vanishes on `plotly_unhover`. You can't study a photo without holding the cursor precisely over its dot. |
| 7 | **No MPS support** | `extract_embeddings.py` only checks for CUDA then falls back to CPU — no Apple Silicon path. |
| 8 | **No arg validation** | Bad CLI inputs produce cryptic Python tracebacks rather than clear usage errors. |

---

## Head-to-Head Comparison

| Dimension | Pipeline 1 (ResNet50) | Pipeline 2 (DINOv3) | Winner |
|---|---|---|---|
| **Semantic quality** | Medium — ImageNet labels | High — visual foundation model | P2 |
| **Setup friction** | Minimal | High (repo + weights) | P1 |
| **UMAP experimentation** | Full re-embed every time | Instant (cached embeddings) | P2 |
| **Visualisation options** | Scatter only | Scatter + Canvas wall | P2 |
| **Browser render performance** | WebGL (`ScatterGL`) | SVG (`scatter`) | P1 |
| **Offline output** | Always | After first CDN load | P1 |
| **Persistent image preview** | Sidebar (persists on unhover) | Floating tooltip (disappears) | P1 |
| **Multi-format input** | JPEG only | 7 formats | P2 |
| **GPU support** | MPS only | CUDA + fp16 | P2 |
| **Subset / prototyping flag** | `--max-images` | None | P1 |
| **Arg validation** | Yes | No | P1 |
| **Large library (5k+ photos)** | Slow, no caching | Efficient | P2 |
| **Modularity / extensibility** | Monolith | Fully decoupled stages | P2 |
| **Touch / mobile** | No | Pinch zoom + pan | P2 |

---

## What Pipeline 1 Has That Pipeline 2 Should Adopt

These are concrete, transferable features — not architectural philosophy.

### 1. `ScatterGL` instead of `scatter` — `plot_viewer.py:157`

One word change. Switches the Plotly trace from SVG to WebGL. Makes the scatter fluid for thousands of points. The current `"type": "scatter"` is a performance regression compared to Pipeline 1's `go.Scattergl`.

### 2. `--max-images` / `--subsample` flag — `extract_embeddings.py`

~10 lines. Enables running the full pipeline on a random subset for rapid iteration. Essential when tuning UMAP hyperparameters or building a new viewer — you don't want to wait on a 5000-image embed for a layout experiment.

### 3. Persistent sidebar — `plot_viewer.py`

~40 lines of HTML/CSS/JS. Replace the disappearing floating overlay with a fixed `<aside>` panel that holds the last hovered image. The scatter takes the left column; the sidebar takes the right. Fundamentally better UX when comparing clusters.

### 4. Inline Plotly option — `plot_viewer.py`

~15 lines. Add `--offline` flag that downloads and inlines the Plotly JS bundle instead of loading from CDN. Makes `photo_map.html` portable to any machine with no network access — consistent with `photo_wall.html` which is already fully offline.

### 5. MPS device support — `extract_embeddings.py`

~10 lines. Mirror Pipeline 1's `resolve_device()` logic. Allows DINOv3 inference on Apple Silicon — not relevant for this WSL setup, but makes the codebase cross-platform.

---

## Recommendation

**Use Pipeline 2 as the primary tool.** The semantic quality difference is decisive. A photo map built on DINOv3 clusters by visual character — colour tone, depth of field, scene geometry, mood. A map built on ResNet50 clusters by ImageNet object classes. For photography work, only DINOv3 produces a map that is actually useful for editing decisions, portfolio curation, or finding visually related shots.

**Keep Pipeline 1** as a lightweight sanity-check tool — useful when the DINOv3 setup is not available or when you want a fast first-pass answer.

**Port the 4 features above** from Pipeline 1 into Pipeline 2. They are low-effort, high-impact, and address the most concrete UX regressions.

**Add an orchestration script** to chain the 4 stages with sensible defaults — this is the biggest usability gap in Pipeline 2 and costs about 30 lines of shell.
