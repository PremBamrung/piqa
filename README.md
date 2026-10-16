# piqa

**P**ython **I**mage **Q**uality **A**ssessment: a local, self-hosted pre-editing engine for
Fuji JPEG/RAF libraries. Sort imports by date, lay a shoot out by visual similarity, and soon:
`piqa choose`.

See [docs/CONTEXT.md](docs/CONTEXT.md) for the vision, decisions and roadmap, and
[docs/PROGRESS.md](docs/PROGRESS.md) for the current milestone.

## Commands

Run from your library root (the folder holding `piqa.toml` and the `Import/` folder that
`ingest` sorts into by default):

```shell
piqa ingest                      # sort the files you copied into Import/ into Import/<date>/<EXT>/
piqa ingest -n                   # dry run: show what would happen
piqa ingest <folder>             # sort another folder's loose files instead of Import/

piqa wall <folder> --open        # embed + layout + wall viewer, then open it
piqa wall <folder> --max-images 200 --color-by top
piqa embed <folder>              # embeddings + thumbnails only

piqa diff <a> <b> -v             # files in <a> missing from <b> (same relative paths)
```

`piqa <command> --help` lists every option (model, batch size, UMAP settings, …).

piqa never reads a memory card: copy the card's files into `Import/` yourself, then run
`piqa ingest`. It moves the loose files sitting directly in `Import/` (subfolders are left
alone), dates them by EXIF capture date (a RAF follows its JPEG; videos use the file date) and
never overwrites. A file already sorted is skipped and reported. When a *different* file
already has the name (two bodies both write `DSCF####`), the incoming file and its RAF/JPG
partner get the camera model as a suffix: `DSCF0412_XE3.JPG`.

`embed`/`wall` write everything into `<folder>/.piqa/`:

| File | What |
|---|---|
| `embeddings-<model>.npz` | DINOv3 vectors; re-runs only embed new or changed files |
| `thumbs/s/`, `thumbs/m/` | 128 px and 512 px thumbnails, made during the same read |
| `coords.npz` | 2-D UMAP coordinates (reused while the embeddings and settings don't change) |
| `grid.npz` | the UMAP layout snapped to a grid (one photo per cell), and the free layout spread so photos stop overlapping once zoomed in |
| `atlas/` | every photo at 64 px packed into a few sprite sheets, for a fast overview |
| `wall.html` | the wall: similar photos side by side; zoom, pan, click for a lightbox |

The wall loads everything from `.piqa/` by relative path, so keep it inside the folder. In the
page: **G** switches grid / free UMAP layout, **P** a preview panel, **/** search, the legend
filters by folder, the minimap jumps around, and the lightbox steps through the 12 most
similar photos with **← →**. **?** lists every key. The view is kept in the URL, so a reload
or a bookmark lands where you were.

## Configuration

`piqa.toml`, looked up from the current directory upwards (paths relative to the file):

```toml
[dinov3]
repo = "dinov3_repo"        # local clone of facebookresearch/dinov3
weights = "dinov3"          # a .pth file, or a folder with <model>_pretrain_*.pth
model = "dinov3_vits16"     # or dinov3_vitb16, dinov3_vitl16, dinov3_vith16plus
```

## Install

Run piqa **natively on Windows or macOS**, not from WSL on `/mnt/<drive>`: the WSL↔Windows
file bridge reads photos ~5× slower (see docs/CONTEXT.md).

Install [uv](https://docs.astral.sh/uv/), then:

```shell
uv tool install --editable <path-to-this-repo>   # puts `piqa` on PATH; code edits apply immediately
piqa --version
```

Windows/Linux get CUDA 12.8 PyTorch (needed for RTX 50xx); macOS gets the standard build
and uses the Apple GPU (MPS).

## Development

```bash
uv sync                    # creates .venv in the repo (from WSL, set UV_PROJECT_ENVIRONMENT
                           # to a path outside the repo so it doesn't clash with Windows')
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
