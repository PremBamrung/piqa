"""Sort camera files into `<dest>/<YYYY-MM-DD>/<EXT>/<file>` (successor of transfert_media.py).

- `piqa ingest` (run from the Photography folder) sorts the loose files sitting directly in
  `Import/`, moving them, exactly like the old script.
- `piqa ingest E:\\DCIM` scans a card recursively and *copies* into `Import/` (never moves off
  a card unless `--move`).

The date is the EXIF capture date (DateTimeOriginal) when available. A RAF shares the date of
its same-name JPEG, or falls back to the EXIF of the JPEG preview embedded in the RAF; videos
and anything else fall back to the file's modification time.
"""

from __future__ import annotations

import datetime as dt
import io
import shutil
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from piqa.images import is_junk

MEDIA_EXTS = {".jpg", ".jpeg", ".raf", ".dng", ".hif", ".heic", ".mp4", ".mov"}
EXIF_IFD = 0x8769
DATETIME_ORIGINAL = 36867
RAF_MAGIC = b"FUJIFILMCCD-RAW"


@dataclass
class Move:
    src: Path
    dst: Path
    date: str


def _date_from_exif_image(img: Image.Image) -> str | None:
    raw = img.getexif().get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL)
    if not raw:
        return None
    try:
        return dt.datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d")
    except ValueError:
        return None


def raf_preview(path: Path) -> bytes | None:
    """The JPEG preview embedded in a Fujifilm RAF (offset/length live at bytes 84-91)."""
    with path.open("rb") as f:
        header = f.read(92)
        if not header.startswith(RAF_MAGIC) or len(header) < 92:
            return None
        offset, length = struct.unpack(">II", header[84:92])
        f.seek(offset)
        return f.read(length)


def capture_date(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext in {".jpg", ".jpeg"}:
            with Image.open(path) as img:
                if date := _date_from_exif_image(img):
                    return date
        elif ext == ".raf" and (preview := raf_preview(path)):
            with Image.open(io.BytesIO(preview)) as img:
                if date := _date_from_exif_image(img):
                    return date
    except Exception:
        pass  # unreadable metadata: fall back to mtime
    return dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def collect(src: Path, recursive: bool) -> list[Path]:
    candidates = src.rglob("*") if recursive else src.iterdir()
    return sorted(
        p
        for p in candidates
        if p.is_file()
        and not is_junk(p.name)
        and p.suffix.lower() in MEDIA_EXTS
        and not any(is_junk(part) for part in p.relative_to(src).parts[:-1])
    )


def plan(src: Path, dest: Path) -> list[Move]:
    """Where every media file under `src` should go. Top-level only when sorting `dest` itself."""
    recursive = src.resolve() != dest.resolve()
    files = collect(src, recursive)

    # JPEG dates first, so a RAF+JPG pair always lands on the same day.
    jpeg_dates = {
        (p.parent, p.stem.upper()): capture_date(p)
        for p in files
        if p.suffix.lower() in {".jpg", ".jpeg"}
    }
    moves = []
    for p in files:
        date = jpeg_dates.get((p.parent, p.stem.upper())) or capture_date(p)
        moves.append(Move(p, dest / date / p.suffix[1:].upper() / p.name, date))
    return moves


def ingest(src: Path, dest: Path, *, copy: bool, dry_run: bool = False) -> None:
    if not src.is_dir():
        raise SystemExit(f"Not a folder: {src}")
    moves = plan(src, dest)
    if not moves:
        print(f"[piqa] nothing to import in {src}")
        return

    verb = "copy" if copy else "move"
    done: Counter[tuple[str, str]] = Counter()
    skipped = 0
    for m in moves:
        if m.dst.exists():
            same = m.dst.stat().st_size == m.src.stat().st_size
            reason = "already imported" if same else "a DIFFERENT file exists"
            print(f"[piqa] skip {m.src.name}: {reason} at {m.dst}")
            skipped += 1
            continue
        if not dry_run:
            m.dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copy2 if copy else shutil.move)(m.src, m.dst)
        done[(m.date, m.dst.parent.name)] += 1

    prefix = "[dry run] would " if dry_run else ""
    for date in sorted({d for d, _ in done}):
        counts = ", ".join(f"{ext} {n}" for (d, ext), n in sorted(done.items()) if d == date)
        print(f"{prefix}{verb} -> {dest / date}: {counts}")
    print(
        f"[piqa] {sum(done.values())} files {'to ' + verb if dry_run else verb + 'd'}"
        f"{f', {skipped} skipped' if skipped else ''}"
    )
