# piqa

**P**ython **I**mage **Q**uality **A**ssessment: a local, self-hosted pre-editing engine for
Fuji JPEG/RAF libraries. Sort cards, map a shoot by visual similarity, and soon: `piqa choose`.

See [docs/CONTEXT.md](docs/CONTEXT.md) for the vision, decisions and roadmap, and
[docs/PROGRESS.md](docs/PROGRESS.md) for the current milestone.

## Commands

Run from your library root (the folder holding `piqa.toml` and the `Import/` folder that
`ingest` sorts into by default):

```shell
piqa ingest                      # sort loose files in Import/ into Import/<date>/<EXT>/ (moves)
piqa ingest <card>               # copy a card or folder into Import/<date>/<EXT>/
piqa ingest <card> -n            # dry run: show what would happen
piqa ingest <card> --dest <dir>  # sort into another root instead of Import/

piqa map <folder> --open         # embed + UMAP + viewers, then open the wall
piqa map <folder> --max-images 200 --color-by top
piqa embed <folder>              # embeddings + thumbnails only

piqa diff <a> <b> -v             # files in <a> missing from <b> (same relative paths)
```

`piqa <command> --help` lists every option (model, batch size, UMAP settings, …).

`ingest` dates files by EXIF capture date (a RAF follows its JPEG; videos use the file date)
and never overwrites: existing files are skipped and reported.

`embed`/`map` write everything into `<folder>/.piqa/`:

| File | What |
|---|---|
| `embeddings-<model>.npz` | DINOv3 vectors; re-runs only embed new or changed files |
| `thumbs/s/`, `thumbs/m/` | 128 px and 512 px thumbnails, made during the same read |
| `coords.npz` | 2-D UMAP coordinates |
| `map.html` | scatter (WebGL); hover to preview, click to open the original |
| `wall.html` | every thumbnail at its UMAP position; zoom, pan, click for a lightbox |

The viewers load thumbnails from `.piqa/thumbs/`, so keep them inside the folder.

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
