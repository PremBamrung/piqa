# Progress

Live checklist for the current milestone. `[x]` done · `[~]` in progress · `[ ]` to do.

## Milestone 1 — Consolidate (2026-10-08)

- [x] Clone renamed GitHub repo (`pix` → `piqa`) into `Photography/piqa`
- [x] Remove the empty `pix` scaffold (app/, Docker, setup.py, …)
- [x] `docs/CONTEXT.md` — vision, blueprint, decisions, measurements
- [x] Copy `ANALYSIS.md`, `UNIFIED_PIPELINE_PLAN.md`, `OPTIMIZATION_ANALYSIS.md` into `docs/`
      (originals left in place)
- [x] `pyproject.toml` (uv, `piqa` command, CUDA 12.8 torch on Windows/Linux, MPS on macOS)
- [x] `images.py` — discovery (skips `._` files), 1/8-scale JPEG decode, thumbnail cache
- [x] `config.py` — `piqa.toml` lookup for DINOv3 repo/weights/model
- [x] `embed.py` — incremental DINOv3 embeddings, one read per image
- [x] `viewers.py` + templates — UMAP, `map.html` (WebGL scatter, persistent preview),
      `wall.html` (zoomable wall, lazy hi-res thumbs, culling, lightbox with original)
- [x] `ingest.py` — successor of `transfert_media.py` (EXIF dates, RAF pairing, card copy)
- [x] `diff.py` — successor of `check_files.py`
- [x] `cli.py` — `piqa ingest | embed | map | diff`
- [x] Tests (ingest, discovery, thumbnails) — 9 passing; ruff clean
- [x] README, `.gitignore`, pre-commit (ruff)
- [x] `piqa.toml` in the Photography folder
- [x] Test in WSL: ingest dry runs on `Import/` and `Import/Audrey` (1,992 files, RAF/JPG
      pairs land on the same day); `piqa map` on 68 Japan photos (CUDA fp16), cached re-run,
      both viewers checked in headless Chrome; `uv tool install --editable` gets CUDA torch
- [x] DINOv3 loads backbones directly (the repo's hubconf.py needs torchmetrics for its heads)
- [x] Docs pass: README `diff` example fixed; CONTEXT.md §5 maps every earlier recommendation
      to its status; older analyses marked historical
- [ ] **You**: Windows setup: install uv, `uv tool install --editable .\piqa`, run on a real folder
- [ ] Benchmark natively on Windows (expect ~70–100 img/s, decode-bound)
- [ ] Commit + push (waiting for your go)

Old scripts in `Photography/` are **kept** until piqa has replaced them in practice.

## Next — Milestone 2: `piqa choose`
- [ ] Burst grouping, eye/face-crop sharpness, InsightFace blinks, pretrained TOPIQ
- [ ] Rating written into JPEGs, rejects moved to `_rejects/`
