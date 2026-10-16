# piqa — Project Context

Consolidated context for **piqa** (Python Image Quality Assessment — and yes, `piqa choose`).
This file gathers the original vision, the technical blueprint, and the decisions and
measurements made while consolidating the earlier projects (`pix`, `pix_old`, and the
Photography scripts) into this repo.

- [1. Vision](#1-vision)
- [2. Technical blueprint](#2-technical-blueprint)
- [3. Decisions & findings (2026-10-08)](#3-decisions--findings-2026-10-08)
- [4. Roadmap](#4-roadmap)
- [5. Earlier plans: what piqa adopted](#5-earlier-plans-what-piqa-adopted)
- [6. Earlier research notes (early 2026)](#6-earlier-research-notes-early-2026)
- Earlier analyses of the pre-piqa scripts (May 2026, historical):
  [ANALYSIS.md](ANALYSIS.md), [UNIFIED_PIPELINE_PLAN.md](UNIFIED_PIPELINE_PLAN.md),
  [OPTIMIZATION_ANALYSIS.md](OPTIMIZATION_ANALYSIS.md)
- Current milestone: [PROGRESS.md](PROGRESS.md)

---

## 1. Vision

At a high level, you are building a custom, self-hosted "pre-editing engine" that automates the tedious, time-consuming grunt work of photo culling and video assembly—allowing you to start your creative process at the 80% mark instead of zero.

Here is what you are ultimately accomplishing:

Automating the "Janitorial" Pass: Eliminating 80% of unusable media—blinks, micro-focus misses, bad composition, dead audio, and bad takes—automatically upon importing your memory cards.

Replacing Black-Box SaaS with Open Source: Replacing $30–$50/month SaaS tools (Aftershoot, Imagen, Descript) with a custom, privacy-focused open-source ML stack that you fully own, tweak, and run locally.

Optimizing Compute Efficiency: Bypassing the hardware bottleneck of slow, VRAM-heavy Vision-LLMs by using hyper-fast, specialized edge vision models (DINOv2/v3, InsightFace, PyTorch TOPIQ) for batch processing at 150+ FPS.

Seamless Workflow Integration: Outputting standard .xmp sidecar files and NLE timeline scripts so Lightroom, Capture One, and DaVinci Resolve natively inherit your AI ratings, flags, and cuts without changing your existing editor.

Building a Searchable Media Brain: Running lightweight VLMs and vector databases (Qdrant) solely on your top 20% "keeper" assets, turning your photo library and video B-roll into a searchable database you can query with natural language ("Fuji golden hour street photo in rain").

Essentially, you are using your engineering skills to build an automated assistant that cleans up the noise, preserves your creative taste, and leaves you with only the best raw material to edit.

---

## 2. Technical blueprint

### Architectural Strategy: The Hybrid Edge/Frontier Split

Consumer GPU and Apple Silicon hardware excel at high-throughput signal processing and feature extraction, but suffer when running heavy local Vision-Language Models (VLMs). The optimal blueprint treats local hardware as a dedicated pipeline for fast computer vision tasks ($150+$ FPS) while delegating high-level reasoning or story synthesis to frontier APIs or small, targeted local scripts.

```
[Fuji Card Import]
        │
        ├──> Fast JPEG Extraction (SOOC / Embedded Preview via rawpy)
        │
        ├──> Stage 1: Local Dual-Branch Engine (Prunes ~80% in seconds)
        │     ├── Global Branch: Frozen DINOv2/v3 + MLP Preference Head
        │     └── Local Branch: InsightFace (Blinks) + Laplacian Crop (Micro-Focus)
        │
        ├──> XMP Sidecar Generation (Write 1–5 Star Ratings directly to disk)
        │
        └──> Stage 2: Semantic Indexing (Top 20% Keepers Only)
              └── VLM Captioning (Florence-2/Qwen) ──> Qdrant Vector Search
```

### 2.1 Fuji RAW+JPG Culling Pipeline

#### I/O & Sensor Optimization

Demosaicing Fujifilm X-Trans `.RAF` files via `LibRaw` is computational overhead ($300\text{ ms/image}$).

* **Zero-Demosaic Rule:** Extract embedded full-resolution preview JPEGs using `rawpy.extract_thumb()` or `exiftool -b -PreviewImage` ($5\text{ ms/image}$) or run directly on SOOC JPEGs.
* Perform all Stage 1 feature extraction on JPEGs, then apply the resulting ratings to both `.RAF` and `.JPG` files via matching `.xmp` sidecars.

#### Stage 1: Dual-Branch Filtering

Vision Transformers downsample patch grids and are naturally blind to micro-focus misses on 26MP/40MP sensors. Stage 1 splits into two lightweight parallel branches:

1. **Global Aesthetic Branch:**
   * **Backbone:** Frozen DINOv2 or DINOv3 (`ViT-L/14`). DINO's self-supervised patch distillation captures composition, lighting, and geometry better than CLIP.
   * **Head:** MLP trained with a Bradley-Terry pairwise preference loss:

   $$\mathcal{L}_{\text{pref}} = -\log \sigma\Big(s(x_a) - s(x_b)\Big)$$

2. **Local Detail Branch:**
   * **Face & Eye Detection:** InsightFace locates facial bounding boxes and calculates Eye Aspect Ratio (EAR) for blink detection.
   * **Micro-Sharpness:** Runs Laplacian Variance or PyTorch TOPIQ on full-resolution $100\%$ crops surrounding detected eyes/faces to flag missed focus (e.g., focus landing on an eyebrow rather than the pupil).

* **Combined Score Formulation:**

  $$\text{Score} = w_1 \cdot S_{\text{DINO}}(x) + w_2 \cdot S_{\text{Focus}}(\text{Crops}) - w_3 \cdot P_{\text{Blink}}$$

#### Stage 2: Semantic Metadata & Vector Search

To conserve memory and compute, run VLMs (Florence-2 or Qwen2.5-VL) **only on the top 20% accepted photos**:

* Generate rich metadata (captions, lighting, dominant colors, scene type) as structured JSON.
* Store text/image embeddings in **Qdrant** or **FAISS** to enable natural language queries across your catalog (*"Fuji golden hour street photo in rain"*).

#### Output Integration

Write scores directly to standard `.xmp` sidecar files using `python-xmp-toolkit` or `pyexiftool`. Naming the sidecar file to match the base file name (e.g., `DSCF1001.xmp`) ensures both `DSCF1001.RAF` and `DSCF1001.JPG` inherit the rating automatically in Lightroom Classic, Capture One, and Darktable.

### 2.2 Custom ML Training & Data Strategy

You do not need to annotate thousands of images manually to train the Stage 1 preference head.

1. **Catalog Mining:** Parse historical Lightroom or Capture One catalogs using `pyexiftool` to extract existing 1–5 star ratings, pick flags, and deleted photos as training targets.
2. **Burst-Pair Hard Negative Mining:**
   * Group photos captured within a $<3\text{-second}$ capture window.
   * Pair a 5-star "keeper" against a 1-star or deleted shot from the **same burst set**.
   * *Why it works:* Because lighting, scene composition, and subject remain identical within a burst, the pairwise loss forces the model head to isolate facial expression, micro-sharpness, and subtle timing rather than macro scene semantics.
3. **Active Learning Loop:** Run unannotated sets through the head, extracting pairs where $\vert s(x_a) - s(x_b)\vert < \epsilon$ (high model uncertainty) for quick manual review.

### 2.3 Self-Hosted Video Automation Stack

| Task | Tool / Component | Execution Strategy |
| --- | --- | --- |
| **Silence Removal** | Silero VAD + `auto-editor` | Strip silences and low-energy audio at the FFmpeg level prior to NLE import. |
| **Transcription & Diarization** | WhisperX (CTranslate2) | Generate word-level timestamps and speaker diarization via `PyAnnote` locally at $>100\times$ real-time speed. |
| **NLE Assembly** | DaVinci Resolve Python API | Parse Whisper/VAD JSON outputs to build timeline tracks, generate paper-cut markers, and color-code clips programmatically. |
| **Audio Enhancement** | DaVinci Voice Isolation | Isolate dialogue and eliminate room reverb locally on GPU. |
| **B-Roll Indexing** | PySceneDetect + Qdrant | Segment video into scenes, extract keyframes, embed via DINO/Florence-2, and index for semantic retrieval. |

---

## 3. Decisions & findings (2026-10-08)

### Actual workflow
- Main camera: Fujifilm X-T3, shooting RAW+JPG. Day-to-day work is **sorting**, mostly on
  SOOC JPEGs, not RAW processing. Lightroom Classic is installed but rarely used.
- Video is a separate story — out of scope for the first versions.
- Photos live on Windows (`G:\Users\Prem\Pictures\Photography`, WD SN570 NVMe). A MacBook
  may also be used (no CUDA → MPS/CPU).
- The repo lives at `Photography/piqa` so `piqa` can be run from the Photography folder,
  like the old `transfert_media.py`.

### Naming
- Repo, package and command: **`piqa`** (renamed from the GitHub repo `pix`).
- Note: `piqa` is also taken on PyPI by an unrelated IQA library (François Rozet). Fine for a
  personal tool that isn't published; don't install that package in the same environment.
  TOPIQ comes from `pyiqa` (IQA-PyTorch), which imports as `pyiqa` and does not clash.
- Fun follow-ups: `piqa choose` (cull), `piqa dex` (natural-language search — a Pokédex).

### Measurements
| What | Result |
|---|---|
| Read 300 X-T3 JPEGs (~16 MB each) from `G:` via WSL `/mnt/g` | 170 MB/s, ~11 files/s |
| Same kind of read natively on Windows | 850 MB/s, ~61 files/s |
| Earlier DINOv3 ViT-S/16 run (1,869 images, WSL) | 7.6 img/s — I/O-bound, not GPU-bound |
| X-T4 RAF embedded preview | 4416×2944 (SOOC JPEG is 6240×4160) |
| `Export/Japon` exports matched to `Import/Japon` originals | 646 of 760 (of 5,353 JPG + 5,353 RAF) |
| `Import/Japon/Laura/selected_files.csv` vs `index.csv` | 148 picks of 2,845 (photos stored elsewhere) |
| Decode one 15 MB, 26 MP Fuji JPEG (one CPU core) | full: ~270 ms · PIL `draft()` at 1/8 scale: ~110 ms |
| Fuji SOOC JPEG embedded previews | only the 160 px EXIF thumbnail (no larger MPF preview) |

Consequences:
- **Run piqa natively on Windows** (or the Mac), not inside WSL: the WSL↔Windows file
  bridge (9p) is the bottleneck, not the NVMe disk or the GPU. WSL is fine for editing code.
- **Decode cost is next**: `draft()` only gives ~2.4×, because decoding the 15 MB of compressed
  data costs ~110 ms even at reduced scale. With 8–12 worker processes that is ~70–100 img/s.
  Beyond that: GPU JPEG decoding (nvJPEG via `torchvision.io.decode_jpeg`). Only the
  sharpness branch needs full-resolution crops.
- **The embedded RAF preview is not full resolution** on Fuji bodies — use the SOOC JPEG for
  micro-focus checks; the preview is fine for the global branch.
- **Training labels already exist**, without catalog mining: Export vs Import (~12% keep rate
  on the Japan trip), and Laura's earlier pick list (148 of 2,845, ~5%). Match on folder +
  capture time rather than the `DSCF` number alone, since numbers repeat across cameras and
  counter resets.

### Corrections to the blueprint
- **150+ FPS** is the model's ceiling, not the pipeline's: reading and decoding are the
  limits. Measure end-to-end.
- **XMP sidecars and JPEGs**: Lightroom ignores sidecar `.xmp` files for JPEGs (it reads
  metadata embedded in the JPEG) and handles RAW+JPEG pairs through the RAW. darktable names
  sidecars `DSCF1001.RAF.xmp`. For a JPEG-first workflow, write the rating **inside the JPEG**
  (Windows Explorer shows it as stars; Lightroom reads it on import); a RAF sidecar is optional.
- **Qdrant**: a personal keeper library fits in a local file index (FAISS / sqlite-vec); a
  vector DB server is unnecessary.
- **Video**: separate dependencies (WhisperX, pyannote, auto-editor). Resolve's external
  scripting and Voice Isolation likely require Resolve Studio — verify before relying on them.

### Design choices
- Each photo folder gets a `.piqa/` cache (embeddings, thumbnails, viewers) with **relative
  paths**, so work is reused across Windows and Mac on the same library.
- `choose` sorts rather than deletes: rejects move to `_rejects/` with their RAF; ratings are
  written into the JPEG.
- Superseded, but **kept on disk until piqa has replaced them in practice**: the scripts in
  `Photography/` (`extract_embeddings.py`, `umap_reduce.py`, `plot_viewer.py`,
  `wall_viewer.py`, `transfert_media.py`, `check_files.py`, `image_map/`), `~/pix_old`
  (ResNet34/t-SNE experiments, plus ~11 GB of copied images to check before removing), and
  the old local clone `~/pix`. The empty `pix` scaffold was removed from this repo.
- `DINOv3` is loaded from the local `dinov3_repo` clone by importing `dinov3.hub.backbones`
  directly; the repo's `hubconf.py` also imports segmentation/text heads that need
  `torchmetrics` and more.

---

## 4. Roadmap

1. **Consolidate** — this repo: `piqa ingest`, `piqa embed`, `piqa wall` (was `piqa map`), `piqa diff`, with the
   `.piqa/` cache and fast decoding.
2. **`piqa choose` without training** — burst grouping, eye/face-crop sharpness, InsightFace
   blinks, pretrained TOPIQ, rating written into JPEGs, rejects moved to `_rejects/`.
3. **Labels & training** — `piqa label` (Export vs Import), `piqa train` (Bradley-Terry head on
   burst pairs). Success criterion: beats pretrained TOPIQ at predicting real exports on the
   Japan set. Pitfalls to avoid: §6.
4. **`piqa dex`** — VLM captions + local vector index on keepers only.
5. **Video** — later, as `piqa video` or a separate project.

---

## 5. Earlier plans: what piqa adopted

Status of the recommendations in the May 2026 analyses, as of milestone 1.

| Recommendation | Source | Status in piqa |
|---|---|---|
| One command for the whole pipeline | UNIFIED_PIPELINE_PLAN | Done: `piqa wall` (embed → UMAP → grid → wall) |
| WebGL scatter (`scattergl`) | ANALYSIS, UNIFIED | Done, then removed 2026-10-16 with `map.html` (the wall replaces it) |
| Persistent preview sidebar | ANALYSIS, UNIFIED | Done; click opens the original |
| Offline Plotly | ANALYSIS, UNIFIED | Removed 2026-10-16 with `map.html` |
| `--max-images` subset | ANALYSIS, UNIFIED | Done |
| MPS (Apple GPU) support | ANALYSIS, UNIFIED | Done: cuda → mps → cpu |
| All image formats, not just JPEG | ANALYSIS | Done (7 extensions) |
| Clear errors instead of tracebacks | ANALYSIS, UNIFIED | Partly: config, weights, model and folder checks |
| Copy photos to WSL ext4 first | OPTIMIZATION (Tier 0) | Replaced: run natively on Windows (measured ~5× faster reads) |
| Skip macOS `._` files | OPTIMIZATION #8 | Done (all hidden files and folders) |
| Thumbnail cache on disk | OPTIMIZATION #7 | Done: 128 px + 512 px tiers, written during the embedding read |
| Viewers re-read every original | OPTIMIZATION | Fixed: viewers load cached thumbnails by relative path (no base64) |
| Incremental embedding cache | OPTIMIZATION #15 | Done: size + mtime fingerprint per file |
| `np.savez` without compression | OPTIMIZATION #4 | Done |
| UMAP `n_jobs=-1` | OPTIMIZATION #1 | Opt-in: `--parallel` (drops the seed, as umap-learn requires) |
| Wall viewport culling | OPTIMIZATION #13 | Done, plus lazy 512 px thumbnails when zoomed in |
| Spatial grid for wall hit-testing | OPTIMIZATION #14 | Not needed yet (linear scan is fine at ~5k images) |
| Reduced-scale JPEG decode | (new) | Done: PIL `draft()`; ~2.4× faster, see §3 |
| GPU JPEG decode (nvJPEG) | OPTIMIZATION #5 | Not yet: the next speed lever if decode-bound |
| `batch_size` 256, GPU-side accumulation, `torch.compile` | OPTIMIZATION #2, #3, #11 | Not yet: the GPU is not the bottleneck |
| pyvips decoding, PCA before UMAP | OPTIMIZATION #6, #9, #10 | Not yet |

---

## 6. Earlier research notes (early 2026)

From a research conversation held before the blueprint. Most of it is superseded by §2–3
(DINO instead of CLIP/MobileNet, InsightFace instead of mediapipe/dlib, ratings inside the
JPEG instead of XMP sidecars). What is still useful:

### Training pitfalls (roadmap step 3)
- **Split by shoot, not by image.** Validation photos must come from shoots the head has
  never seen; a random split of one trip lets it memorize the place and the light. On the
  Japan set, hold out whole days or folders, and keep both frames of a burst pair on the
  same side.
- **Class imbalance.** Keep rates are ~5–12% (§3), so a classifier that always says "reject"
  scores ~90% accuracy. Bradley-Terry pairs avoid most of this. A pointwise head needs a
  balanced sampler, and should be judged on keeper precision/recall or AUC, not accuracy.
- **Noisy labels.** "Not exported" does not mean bad: it can be a near-duplicate of a pick,
  or off-story. Within-burst pairs are the cleanest signal; cross-burst rejects are weak
  negatives.

### Culling behavior
- **Intentional "flaws".** Grain, intentional camera movement and low-key B&W get punished
  by sharpness and IQA models. Send low-confidence calls to a review bucket rather than
  `_rejects/`, and let the global branch overrule a low sharpness score.
- **Sharp where it matters.** Face crops cover people. Street, landscape and travel frames
  without a face need a subject region (saliency, a SAM mask, or DINO patch attention)
  before measuring sharpness; otherwise a sharp background passes.
- **Best expression in a burst.** Expression probabilities (FER models) could break ties
  within a burst. Low priority: the burst-pair head should learn this on its own.

### Other label sources (if Export vs Import is not enough)
- **Commercial culler as teacher.** Run Aftershoot, Narrative Select or FilterPixel (trials)
  on archived shoots, then parse the ratings/labels they write to XMP into training targets.
  It learns their taste, not yours, and their misses become your model's misses: check
  their keepers by hand, and check each tool's terms of use.
- **VLM as teacher.** A frontier VLM scores a few hundred photos (composition, light,
  emotion, keep/reject, a one-line reason) and a head is trained on those scores. This sends
  photos to an API, which goes against the local-first goal, and VLMs are weak at low-level
  quality (see Q-Bench below): use it for aesthetics and semantics only, never focus or blur.

Both are secondary to your own picks: useful for pretraining or for seeding the active
learning loop.

### Research & datasets
| Item | What | Relevance to piqa |
|---|---|---|
| [Q-Bench](https://arxiv.org/abs/2309.14181) (ICLR 2024) | Benchmark of MLLMs on low-level perception | They are poor at blur and noise: keep focus and blinks in the local branch |
| Q-Instruct (CVPR 2024) | ~200k low-level visual instruction pairs | Only if a VLM is ever fine-tuned for quality |
| [Q-Bench-Portrait](https://arxiv.org/abs/2601.18346) (Jan 2026) | Portrait quality benchmark, 25 MLLMs | Best models still well short of human judgment |
| [HumanAesExpert](https://arxiv.org/abs/2503.23907) (2025) | VLM + HumanBeauty dataset (108k human images, 12 aesthetic dimensions) | Candidate pretrained scorer for people photos |
| AVA | 250k images, crowd 1–10 ratings | Generic aesthetic pretraining or a sanity baseline |
| [PARA](https://arxiv.org/abs/2203.16754) (CVPR 2022) | 31k images, 438 raters, per-rater scores and attributes | Personalized aesthetics, the closest to the "my taste" problem |
| NIMA | CNN predicting the AVA rating distribution | Baseline for the DINO head, next to TOPIQ |
| imagededup | pHash / CNN duplicate finder | Baseline for burst grouping (vs capture time + DINO) |

Dropped from that conversation: Depth Pro as a sharpness model (it is a depth estimator),
mediapipe/dlib eye landmarks (InsightFace chosen), and RAW embedded previews for focus checks
(not full resolution on Fuji, §3).
