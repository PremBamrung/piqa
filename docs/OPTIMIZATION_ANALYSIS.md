# Pipeline Optimization Analysis
> Hardware: RTX 5070 Ti / R7 5800X / 64GB DDR4 / WSL → Windows drive

> **Historical.** Written in May 2026 for the standalone scripts that piqa replaced. Script names, flags and line numbers refer to those scripts. What piqa adopted: [CONTEXT.md §5](CONTEXT.md#5-earlier-plans-what-piqa-adopted).


---

## Empirical Benchmarks (1,869 images, ViT-S/16, fp16)

Real-world run logs from 2026-05-03:

```
STAGE 1 (extract_embeddings):   246.6s total  →  7.6 img/s
  ├─ 1,869 JPEGs found, 2 skipped (._ macOS resource forks)
  ├─ batch_size=64, num_workers=8, fp16
  └─ Output: embeddings.npz (1869, 384) float32

STAGE 2 (umap_reduce):            8.2s total  →  228 img/s
  ├─ n_neighbors=15, min_dist=0.1, metric=cosine
  ├─ ⚠️  n_jobs forced to 1 by random_state=42 (see note below)
  └─ Output: umap_coords.npz

STAGE 4 (wall_viewer):          143.9s total  →   13 img/s
  ├─ thumb_size=500, threads=32
  └─ Output: photo_wall.html (43.8 MB)
```

**Projected at 10K images (linear extrapolation):**
| Stage | 1,869 images | 10K images (est.) | Bottleneck |
|---|---|---|---|
| Stage 1 — Embed | 246s (4.1 min) | **22 min** | WSL 9p I/O + PIL decode |
| Stage 2 — UMAP | 8.2s | **40-50s** (with n_jobs=1) | Single-threaded kNN |
| Stage 3 — Scatter | — | **10-15 min** | Image re-read |
| Stage 4 — Wall | 144s (2.4 min) | **13 min** | Image re-read |
| **Total** | ~7 min | **~47 min** | — |

**Key insight:** At 7.6 img/s, the RTX 5070 Ti is processing images at <1% of its capability. The GPU forward pass for a 256×256 ViT-S/16 takes ~0.3ms per image — meaning the GPU could theoretically handle **3,000+ img/s**. The bottleneck is entirely upstream.

---

## Additional Finding: macOS AppleDouble files polluting glob

The logs show two `._DSCF*` files being picked up by `rglob` and then skipped:

```
[WARN]  Skipping ._DSCF0031.JPG: cannot identify image file '.../._DSCF0031.JPG'
[WARN]  Skipping ._DSCF0373.JPG: cannot identify image file '.../._DSCF0373.JPG'
```

These are macOS resource fork files created when the SD card was accessed on a Mac. They waste I/O time (PIL tries to open them, fails), clutter logs, and would bloat an incremental cache. A simple `._` prefix filter at glob time costs zero runtime and avoids these entirely.

---

## UMAP `random_state` vs `n_jobs` Conflict (confirmed)

The umap-learn library warns:

```
n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
```

You **cannot** have both `random_state=42` AND `n_jobs=-1` with the current umap-learn (0.5.x). The NNDescent kNN graph construction uses random subsampling for approximate nearest neighbors — parallelizing it produces non-deterministic graph ordering, which breaks `random_state` reproducibility.

**Precedent:** For large datasets (5-10K+) where UMAP runtime becomes significant, drop `--random_state` (or set to `None`) to unlock multi-core parallelism. For small datasets (<2K) like this run, 8.2s is already fast enough that reproducibility is worth the single-threaded cost.

**Alternative:** Use `--metric euclidean` directly (skip L2-norm) plus `n_jobs=-1` — euclidean metric doesn't use NNDescent for small datasets, so parallelism works even with a fixed seed in some versions. Or wait for a future umap-learn release that addresses this.

---

## Current Pipeline Flow & Time Profile (10K images estimate)

```
STAGE 1 (extract_embeddings):     I/O-dominated, GPU starved
  ├─ Disk → 10K random reads via WSL 9p          ████████████████  ~70-90%
  ├─ PIL decode + PadToSquare + Resize (CPU)      ███              ~10-15%
  ├─ GPU DINOv3 forward (batch=64)                █                ~2-5%
  └─ np.savez_compressed                          █                ~1-2%

STAGE 2 (umap_reduce):            CPU, single-threaded by default
  ├─ L2-normalize                                 trivial
  ├─ UMAP kNN graph (NNDescent)                   ████████████     ~95%
  │   └─ ❌ n_jobs not set → defaults to 1 core
  └─ np.savez_compressed                          █                ~1-2%

STAGE 3 (plot_viewer):            I/O-dominated (re-reads all images)
  ├─ Disk → re-read 10K images (WSL 9p)           ████████████████ ~80-90%
  └─ Thumbnail encode (ThreadPoolExecutor)        ███              ~10-15%

STAGE 4 (wall_viewer):            I/O-dominated (re-reads all images AGAIN)
  ├─ Disk → re-read 10K images (WSL 9p)           ████████████████ ~80-90%
  └─ Thumbnail encode (ThreadPoolExecutor)        ███              ~10-15%
```

**The WSL 9p filesystem protocol is the dominant bottleneck in Stages 1, 3, and 4.**

---

## Tier 0 — Zero Code Changes, Maximum ROI

### 0. Copy images to WSL's native ext4 before processing

**Effort:** 0 code lines, 1 shell command per run  
**Gain:** **3-10x I/O speed — eliminates the 9p bottleneck entirely for all stages**  
**Why:** WSL accesses Windows drives (`/mnt/g/...`) via the 9p network protocol. Every `Image.open()`, every `path.read_bytes()`, every thumbnail read crosses this boundary with ~50-200ms latency per file. WSL's own root filesystem (`/home/`, `/tmp/`) is native ext4 on a virtual disk — comparable to bare-metal Linux I/O.

By copying images into WSL first, the pipeline never touches 9p. No code changes needed in any stage script. Stages 1, 3, and 4 all benefit without modification.

```bash
# One-time copy before running the pipeline
SRC="/mnt/g/Users/Prem/Pictures/Photography/Source/Laura/Atacama Uyuni/Prem"
DST="$HOME/photo_cache/atacama_prem"

cp -r "$SRC" "$DST"      # or rsync -a for incremental
python extract_embeddings.py --images_dir "$DST" ...
python umap_reduce.py ...
python wall_viewer.py --images_dir "$DST" ...
```

The source has date-organized subfolders with JPGs inside. Here are your options:
Simplest — copy everything including non-JPEG (RAF, MOV, etc.):
mkdir -p ~/photo_cache/atacama_prem
rsync -ah --progress "/mnt/g/Users/Prem/Pictures/Photography/Source/Laura/Atacama Uyuni/Prem/" ~/photo_cache/atacama_prem/
The slowness comes from copying RAF/MOV files you don't need for the pipeline.
JPEG-only — skip the rest:
mkdir -p ~/photo_cache/atacama_prem
rsync -ahm --progress \
  --include='*/' --include='*.JPG' --include='*.jpg' --include='*.jpeg' \
  --exclude='*' \
  "/mnt/g/Users/Prem/Pictures/Photography/Source/Laura/Atacama Uyuni/Prem/" \
  ~/photo_cache/atacama_prem/
-m prunes empty dirs, --include='*/' matches directories so rsync descends into them, then only JPGs are matched.
One-liner if your tour dirs follow the YYYY-MM-DD/JPG/ pattern:
mkdir -p ~/photo_cache/atacama_prem
find "/mnt/g/Users/Prem/Pictures/Photography/Source/Laura/Atacama Uyuni/Prem" \
  -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) \
  -exec cp --parents -t ~/photo_cache/atacama_prem {} +
cp --parents preserves the relative path structure (2026-03-29/JPG/DSCF0001.JPG), which the pipeline expects since it stores relative paths.
All three produce identical output for extract_embeddings.py. The pipeline then runs against ~/photo_cache/atacama_prem on ext4, never touching /mnt/g/.



**Storage / RAM consideration:**

| Location | Type | Speed | Note |
|---|---|---|---|
| `/home/prem/photo_cache/` | ext4 on WSL virtual disk | Fast | Recommended — persistent, survives reboots |
| `/tmp/` | May be tmpfs (RAM) | Fastest | ⚠️ Check `df -h /tmp` first. 10K photos × 15MB = 150GB — won't fit in tmpfs with 64GB RAM |
| `/mnt/g/...` | 9p → NTFS | Slow | Current setup — the bottleneck |

**Copy cost vs savings (1,869 images benchmark):**

```
cp -r (one-time):       ~30-90 seconds
Stage 1 (currently):    246s → ext4 estimate: 30-60s     (4-8x faster)
Stage 3/4 (currently):  144s → ext4 estimate: 15-30s     (5-10x faster)
```

For 10K images: copy cost ~3-8 minutes, savings ~20-35 minutes per full pipeline run. Net positive on the second run.

**Layering strategy — combine with code-level optimizations for multiplicative gains:**

```
┌─ Tier 0: copy to ext4        ← immediate, zero code, eliminates 9p
├─ Tier 1: tqdm + batch_size   ← trivial code wins
├─ Tier 1: UMAP n_jobs=-1      ← massive CPU speedup
├─ Tier 1: GPU JPEG decode     ← removes remaining CPU I/O bottleneck
└─ Tier 1: thumbnail caching   ← eliminates Stage 3/4 entirely
```

After combining Tier 0 with GPU JPEG decode (item 5), the pipeline becomes I/O-on-ext4 → GPU-decode → GPU-inference → GPU-transfer → done. Zero CPU involvement in the critical path for JPEG images.




---

## Tier 1 — High Impact, Low Effort

### 1. Add `n_jobs=-1` to UMAP constructor
**File:** `umap_reduce.py:81`  
**Effort:** 1 line  
**Gain:** **8-12x faster Stage 2**  
**Why:** UMAP's kNN graph construction (NNDescent via numba) defaults to 1 thread. R7 5800X has 16 threads. For 10K × 384-d data, the kNN phase dominates UMAP runtime. Currently it uses 1/16th of the CPU.

```python
# Before
reducer = umap.UMAP(
    n_components=args.n_components,
    ...
)

# After
reducer = umap.UMAP(
    n_components=args.n_components,
    ...
    n_jobs=-1,  # ← use all CPU cores
)
```

---

### 2. Increase batch_size default / GPU utilization
**File:** `extract_embeddings.py:124`  
**Effort:** 1 line (change default)  
**Gain:** **2-4x faster GPU portion of Stage 1**  
**Why:** At fp16, ViT-S+/16 uses ~58MB VRAM. Each 256×256 image is 384KB. Batch=64 uses ~82MB / 16GB — only 0.5% utilization. The RTX 5070 Ti can comfortably handle batch=256-512 for fp16 inference. Every launch of a GPU kernel has overhead; larger batches amortize that overhead across more images.

```python
# Before
parser.add_argument("--batch_size", type=int, default=64)

# After
parser.add_argument("--batch_size", type=int, default=256)
```

**Note:** At batch=512 with fp16, VRAM is ~256MB — still only 1.6% of 16GB. You could go higher but DataLoader memory and Python overhead become the limiter before VRAM.

---

### 3. Accumulate embeddings on GPU, transfer once
**File:** `extract_embeddings.py:176-181`  
**Effort:** ~5 lines  
**Gain:** **15-30% faster per-batch**  
**Why:** Currently each embedding is transferred GPU→CPU individually with `.cpu().float().numpy()`, causing hundreds of small PCIe synchronization events. Accumulate a tensor list on GPU and transfer in bulk.

```python
# Before
all_embeddings = []
...
for emb, path, ok in zip(features, paths, oks):
    if ok:
        all_embeddings.append(emb.cpu().float().numpy())
        all_paths.append(path)

# After
all_embeddings = []  # store tensor on GPU, not numpy
...
features = model(imgs)  # (B, D) on GPU
mask = torch.tensor([ok for ok in oks], device=device)
valid = features[mask]  # filter on GPU
all_embeddings.append(valid)  # keep on GPU
all_paths.extend(p for p, ok in zip(paths, oks) if ok)
...
# At end, single bulk transfer:
embeddings_np = torch.cat(all_embeddings).cpu().numpy().astype(np.float32)
```

---

### 4. Use `np.savez` instead of `np.savez_compressed`
**Files:** `extract_embeddings.py:194`, `umap_reduce.py:103`  
**Effort:** 2 changes (remove `_compressed` suffix)  
**Gain:** **20-40% faster save time**  
**Why:** `savez_compressed` runs deflate (ZIP) compression on the numpy arrays. For float32 embedding matrices (10K × 384 = ~15MB), the compression ratio is poor (maybe 5-10%) and the CPU cost is high. `savez` just writes raw bytes without compression — nearly instant. The disk space difference is negligible for your data sizes.

```python
# Before
np.savez_compressed(args.output, ...)

# After
np.savez(args.output, ...)
```

---

### 5. GPU JPEG decode via nvJPEG / torchvision
**File:** `extract_embeddings.py:93-103` (ImageFolder.__getitem__)  
**Effort:** ~20 lines  
**Gain:** **3-5x faster image decode (eliminates CPU PIL roundtrip)**  
**Why:** PIL decodes JPEG on CPU, creates a PIL Image, converts to RGB, then transforms to tensor and sends to GPU. nvJPEG (via `torchvision.io.decode_jpeg`) decodes directly to a GPU tensor, eliminating the CPU decode and the CPU→GPU transfer. In WSL, this also reduces the 9p protocol overhead since only compressed JPEG bytes cross the boundary, not decoded pixel arrays.

**Caveat:** Only works for JPEG (.jpg, .jpeg). Fall back to PIL for PNG/TIFF/WebP/BMP.

```python
from torchvision.io import decode_jpeg, ImageReadMode


def __getitem__(self, idx):
    path = self.paths[idx]
    try:
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            # GPU decode path — nvJPEG
            data = path.read_bytes()  # fast raw read (no decode)
            tensor = decode_jpeg(
                torch.frombuffer(bytearray(data), dtype=torch.uint8),
                device=self.device,
                mode=ImageReadMode.RGB,
            )
            # Apply PadToSquare-like logic on GPU tensor
            # (or use torchvision.transforms on GPU tensor)
        else:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)  # to CPU tensor, then moved to GPU
        ok = True
    except Exception as exc:
        tensor = torch.zeros(3, 256, 256)
        ok = False
    return tensor, str(path.relative_to(self.root)), ok
```

**Full GPU path:** If you implement GPU padding and resize transforms (via `torchvision.transforms.functional` or custom CUDA kernels), the entire pipeline from disk bytes to DINOv3 output stays on GPU — zero CPU involvement for JPEGs. This is the single biggest win for your WSL setup.

---

### 6. Replace PIL with pyvips for non-JPEG decode
**File:** `extract_embeddings.py:93-103`  
**Effort:** ~10 lines  
**Gain:** **4-10x faster decode for PNG/TIFF/WebP**  
**Why:** pyvips is a streaming image processing library written in C. It decodes, resizes, and reformats in a single pipeline without decompressing to full pixel buffers. Significantly faster than PIL especially for large TIFF/DNG files.

```python
# pip install pyvips
import pyvips


def load_image_pyvips(path: Path, size: int = 256) -> np.ndarray:
    """Load and resize in one pass, returns numpy (H, W, 3) uint8."""
    img = pyvips.Image.thumbnail(str(path), size, height=size, size="force")
    return img.numpy()  # RGB, will be converted to tensor
```

---

### 7. Cache thumbnails as disk files — generate once during Stage 1
**Files:** `extract_embeddings.py` (generate) + `plot_viewer.py` / `wall_viewer.py` (consume)  
**Effort:** ~40 lines  
**Gain:** **Eliminates Stage 3+4 image re-read entirely (confirmed: 144s for 1,869 images)**  

#### Why disk files beat embedding in `.npz`

The earlier `.npz` embedding proposal (flush thumbnails into the numpy archive) is simple but locks you to one size and bloats a data file with binary blobs. Writing thumbnails as real JPEG files in a mirrored directory tree is better:

| Approach | Lock-in | Multiple viewers | Reusable outside pipeline? |
|---|---|---|---|
| Embed in `.npz` | Single size forever | One thumbnail per image | No |
| **Disk files in mirrored tree** | Any viewer picks its own resize | Trivial | Yes — any image tool |

#### Generation: post-inference, single-threaded, idempotent

Don't generate inside `ImageFolder.__getitem__` — DataLoader multiprocessing means workers will race on `mkdir` and file writes. Instead, do it in the main thread after inference, when `all_paths` is available. Check `if thumb_path.exists()` so it's a no-op on subsequent runs.

```python
# ── Place at end of extract_embeddings.py main(), after the inference loop ──
import tqdm

THUMB_SIZE = 300  # 256-300 covers both wall (96) and plot (200) viewers cleanly
THUMB_QUALITY = 75

thumb_dir = args.output.parent / "thumbs"
print(f"[INFO]  Generating thumbnails → {thumb_dir} ...")

for path in tqdm.tqdm(all_paths):
    thumb_path = thumb_dir / path
    if thumb_path.exists():
        continue  # ← idempotent — skip on re-runs
    thumb_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(args.images_dir / path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        img.save(str(thumb_path), format="JPEG", quality=THUMB_QUALITY, optimize=True)
```

**Resulting directory structure (mirrors source):**
```
~/photo_cache/atacama_prem/               ← Tier 0 copy (originals, ext4)
  2026-03-29/JPG/DSCF0001.JPG
  2026-03-29/JPG/DSCF0002.JPG

~/photo_cache/thumbs/                     ← generated once by Stage 1
  2026-03-29/JPG/DSCF0001.JPG             ← 300px max, ~12-20KB
  2026-03-29/JPG/DSCF0002.JPG
```

#### Consumption: Stage 3/4 swap to thumbnails dir

Since the relative paths match, viewers just need a `--thumbnails_dir` argument. When provided, they read the small JPEG instead of decoding the full original:

```python
# wall_viewer.py / plot_viewer.py — encode_thumbnail() change
def encode_thumbnail(img_path: Path, size: int, thumb_cache: Path | None = None) -> str:
    """If thumb_cache is provided, read pre-generated thumbnail; fall back to original."""
    src = (thumb_cache / img_path.relative_to(images_dir)) if thumb_cache else img_path
    try:
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img.thumbnail((size, size), Image.LANCZOS)  # 300→96 is near-instant
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=72, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""
```

Reading a 15KB thumbnail from ext4 is ~0.3ms per file vs ~50-200ms for a full 15MB original via 9p — a **200-600x speedup** for the I/O portion. The 144s wall_viewer run drops to ~1-2 seconds.

#### One-time cost vs perpetual savings

```
One-time cost (Stage 1, first run only):
  Generate 10K thumbnails via PIL thumbnail + JPEG encode  ~50-90 seconds (ext4)

Savings per subsequent pipeline run:
  Stage 3 plot_viewer:   15 min → 1-2 sec     (thumbnail read vs original read)
  Stage 4 wall_viewer:   13 min → 1-2 sec     (thumbnail read vs original read)
                        ─────
                         28 min saved per run
```

Combined with Tier 0 (images already on ext4), the first-run thumbnail generation is fast. Every run after that — including UMAP hyperparameter experiments — skips the thumbnail step entirely.

**Note on thumbnail size:** Pick 256-300. That's sharp enough for the 200px plot_viewer hover panel and trivially downscaled to 96px for the wall_viewer. At 300px JPEG quality 75, 10K images use ~150-200 MB on disk — negligible on ext4.

---

### 8. Filter macOS AppleDouble `._` files at glob time
**File:** `extract_embeddings.py:82-84` (ImageFolder.__init__)  
**Effort:** 1 line  
**Gain:** **Eliminates wasted I/O on garbage files**  
**Why:** Logs show `._DSCF0031.JPG` being picked up by `rglob` and failing PIL decode. These are macOS resource forks (AppleDouble format) — zero-byte metadata files with image extensions. PIL tries to open them, fails with "cannot identify image file", and they pollute the log. For SD cards touched by macOS, these can be 1-5% of files in a directory — pure wasted I/O in WSL.

```python
# Before
self.paths = sorted(p for p in root_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)

# After
self.paths = sorted(
    p
    for p in root_dir.rglob("*")
    if p.suffix.lower() in SUPPORTED_EXTS
    and not p.name.startswith("._")  # ← skip macOS AppleDouble forks
)
```

---

## Tier 2 — Medium Impact, Medium Effort

### 9. PCA pre-reduction before UMAP
**File:** `umap_reduce.py` (between L2-norm and UMAP)  
**Effort:** ~10 lines  
**Gain:** **3-5x faster UMAP Stage 2**  
**Why:** DINOv3 embeddings (384-1024-d) have significant redundancy. PCA to 50-100 dimensions preserves >95% variance while giving UMAP's NNDescent a 4-7x smaller search space. The kNN graph construction cost is O(D × N log N), so reducing D linearly speeds up this phase. For 384-d ViT-S, reduce to 50-d — UMAP runs 7x faster.

```python
from sklearn.decomposition import PCA

if args.pca_dims and args.pca_dims < embeddings.shape[1]:
    print(f"[INFO]  PCA: {embeddings.shape[1]} → {args.pca_dims}")
    embeddings = PCA(n_components=args.pca_dims, random_state=args.random_state).fit_transform(
        embeddings
    )
```

---

### 10. PyVIPS for thumbnail encoding in Stages 3/4
**Files:** `wall_viewer.py:40-55`, `plot_viewer.py:44-57`  
**Effort:** ~15 lines per file  
**Gain:** **4-10x faster thumbnail generation**  
**Why:** pyvips can decode, resize, and re-encode a JPEG in a single streaming pass. PIL does full decode → resize in memory → re-encode, much slower. For 10K thumbnails, this turns minutes into seconds.

```python
# Before (PIL)
with Image.open(img_path) as img:
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

# After (pyvips)
img = pyvips.Image.thumbnail(str(img_path), size, height=size, size="down", auto_rotate=True)
buf = img.jpegsave_buffer(Q=72)  # returns bytes directly
b64 = base64.b64encode(buf).decode("ascii")
```

---

### 11. `torch.compile` on DINOv3 model
**File:** `extract_embeddings.py:146`  
**Effort:** 2 lines  
**Gain:** **15-30% faster GPU inference**  
**Why:** `torch.compile()` fuses PyTorch operations into optimized CUDA kernels, reduces Python overhead, and eliminates intermediate tensor allocations. DINOv3 ViT forward has many small operations (attention, layernorm, MLP) that benefit from fusion.

```python
# After model.to(device)
model = model.eval().to(device)
if hasattr(torch, "compile"):
    model = torch.compile(model, mode="reduce-overhead")  # ←
```

**Caveat:** First batch compiles the model (~5-10s for ViT). Subsequent batches run faster. Use `mode="reduce-overhead"` for inference workloads, or `mode="max-autotune"` for CUDA graphs (higher compile time, better throughput).

---

### 12. Pin-memory + prefetch tuning
**File:** `extract_embeddings.py:155-162`  
**Effort:** 3 lines  
**Gain:** **10-20% better DataLoader throughput**  
**Why:** `prefetch_factor=2` means only 2 batches are pre-loaded. Increasing to 4-6 keeps the GPU fed during spikes of slow WSL I/O. The memory cost is minimal (each batch is ~24-48MB).

```python
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=(device.type == "cuda"),
    prefetch_factor=4 if args.num_workers > 0 else None,  # ← was 2
    persistent_workers=True,  # ← avoids worker respawn overhead
    collate_fn=collate_fn,
)
```

---

## Tier 3 — UX / Browser Performance

### 13. Canvas viewport culling (wall_viewer.js)
**File:** `wall_viewer.py` (JavaScript `draw()` function)  
**Effort:** ~20 lines JS  
**Gain:** **Fluid 60fps pan/zoom at 10K images**  
**Why:** Currently `draw()` renders every image in the dataset on every frame — O(N) draw calls. For 10K images, this kills framerate during pan/zoom. Only render images whose bounding boxes intersect the viewport.

```javascript
function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cw = canvas.width, ch = canvas.height;
    for (let i = 0; i < N; i++) {
        const d = IMAGES[i];
        const px = offsetX + d.x * scale;
        const py = offsetY + d.y * scale;
        const { w, h } = imgDisplaySize(d.w, d.h);
        // Viewport culling
        if (px + w/2 < 0 || px - w/2 > cw || py + h/2 < 0 || py - h/2 > ch) continue;
        // ... rest of draw logic
    }
}
```

---

### 14. Spatial index for hit testing (wall_viewer.js)
**File:** `wall_viewer.py` (JavaScript `hitTest()` function)  
**Effort:** ~15 lines JS  
**Gain:** **Eliminates mousemove lag at 10K images**  
**Why:** `hitTest()` iterates ALL images on every mousemove event (100+ times/sec). For 10K images, this is a heavy loop. A simple spatial grid (50×50 cells) reduces lookup to ~4 images.

```javascript
// Build spatial grid once
const GRID = 50; // 50x50 = 2500 cells
const grid = Array.from({length: GRID}, () => Array.from({length: GRID}, () => []));
IMAGES.forEach((d, i) => {
    const gx = Math.floor(d.x * GRID);
    const gy = Math.floor(d.y * GRID);
    if (gx >= 0 && gx < GRID && gy >= 0 && gy < GRID) {
        grid[gy][gx].push(i);
    }
});

function hitTest(mx, my) {
    // Map mouse to UMAP space, then to grid cells to check
    const ux = (mx - offsetX) / scale;
    const uy = (my - offsetY) / scale;
    const gx = Math.floor(ux * GRID);
    const gy = Math.floor(uy * GRID);
    // Check 3x3 neighborhood of cells
    for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
            const nx = gx + dx, ny = gy + dy;
            if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID) continue;
            for (const i of grid[ny][nx]) {
                // ... existing per-image hit test ...
            }
        }
    }
}
```

---

### 15. Incremental embedding cache
**File:** `extract_embeddings.py`  
**Effort:** ~30 lines  
**Gain:** **Near-zero cost for repeated runs on same folder**  
**Why:** If you add/remove a few images from a 10K folder, re-embedding everything is wasteful. Hash each image file (path + mtime + size), only re-embed changed ones, merge results.

```python
import hashlib


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.md5(f"{path.name}:{stat.st_mtime}:{stat.st_size}".encode()).hexdigest()


# On run:
# 1. Scan all images, compute fingerprints
# 2. Load existing embeddings.npz if present
# 3. Identify new/changed fingerprints
# 4. Only embed new/changed images
# 5. Merge with existing embeddings (matching paths)
# 6. Save updated embeddings.npz
```

---

## Recommended Implementation Sequence

Implement in this order for maximum cumulative impact:

```
0. Copy images to WSL ext4 (~/photo_cache/)  ← ⭐ ZERO CODE CHANGES, immediate 4-10x I/O win
1. GPU JPEG decode (nvJPEG)          ← BIGGEST WIN for WSL I/O
2. Thumbnail cache as disk files     ← eliminates Stage 3+4 re-read (confirmed: 144s for 1.9K images; one-time cost, perpetual win)
3. Filter ._ macOS files at glob      ← trivial, prevents wasted I/O on AppleDouble garbage
4. n_jobs=-1 for UMAP (drop seed)     ← trivial, massive UMAP speedup
                                       ⚠️ Must drop --random_state for parallelism
5. Accumulate on GPU, transfer once   ← clean batching fix
6. Increase batch_size to 256         ← GPU utilization (currently 7.6 img/s → target 100+)
7. np.savez (drop compression)        ← trivial save speedup
8. pyvips thumbnails (if caching incomplete)
9. PCA pre-reduction                  ← faster UMAP tuning
10. torch.compile model               ← GPU inference speedup
11. Canvas viewport culling           ← browser UX
12. Spatial hit-test grid             ← browser UX
13. Incremental cache                 ← repeated runs
```

**Estimated cumulative speedup:** 5-20x end-to-end for 10K images on WSL, bringing total runtime from **~47 minutes** (empirically projected) down to **2-8 minutes**.

---

## What NOT to Optimize

- **Model architecture** — DINOv3 ViT-S+/16 is already the sweet spot (384-d, lightweight). ViT-S would be marginally faster but lower quality. ViT-B/16 would be slower with diminishing returns.
- **UMAP metric** — Cosine via L2-norm trick is already optimal. Native cosine in umap-learn is slower due to dynamic dispatch.
- **Thread count for thumbnail encoding** — Already reasonable (cpu_count × 2, capped at 32). The bottleneck is WSL I/O, not CPU threads.
- **HTML output format** — Already efficient. Base64 inline is fine for 10K images (~30MB wall or ~20MB scatter). Could add lazy JS loading for massive datasets but not needed at this scale.
