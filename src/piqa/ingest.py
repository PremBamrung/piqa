"""Sort the files copied into `Import/` into `Import/<YYYY-MM-DD>/<EXT>/<file>`.

Successor of transfert_media.py. You copy a card's files into `Import/` by hand; `piqa ingest`
moves the loose files sitting directly in it (subfolders are left alone). piqa never reads a
card.

The date is the EXIF capture date (DateTimeOriginal) when available. A RAF shares the date of
its same-name JPEG, or falls back to the EXIF of the JPEG preview embedded in the RAF; videos
and anything else fall back to the file's modification time.

Two bodies write the same `DSCF####` names. When a different file already holds the name, the
incoming file gets the camera model as a suffix (`DSCF0412_XE3.JPG`), and so does its RAF/JPG
partner, so the pair stays matched; `_2`, `_3`, … if that is taken too or there is no model.
"""

from __future__ import annotations

import datetime as dt
import io
import re
import shutil
import struct
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from piqa.images import is_junk

MEDIA_EXTS = {".jpg", ".jpeg", ".raf", ".dng", ".hif", ".heic", ".mp4", ".mov"}
EXIF_IFD = 0x8769
DATETIME_ORIGINAL = 36867
MODEL = 0x0110
RAF_MAGIC = b"FUJIFILMCCD-RAW"


@dataclass
class Move:
    src: Path
    dst: Path
    date: str
    already: bool = False  # the same file is already at `dst`


def _date_from_exif(exif: Image.Exif) -> str | None:
    raw = exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL)
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


def read_exif(path: Path) -> Image.Exif | None:
    """EXIF of a JPEG, or of the JPEG preview inside a RAF; None for anything else."""
    ext = path.suffix.lower()
    try:
        if ext in {".jpg", ".jpeg"}:
            with Image.open(path) as img:
                return img.getexif()
        if ext == ".raf" and (preview := raf_preview(path)):
            with Image.open(io.BytesIO(preview)) as img:
                return img.getexif()
    except Exception:
        pass  # unreadable metadata
    return None


def capture_date(path: Path) -> str:
    exif = read_exif(path)
    if exif is not None and (date := _date_from_exif(exif)):
        return date
    return dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def camera_tag(path: Path) -> str | None:
    """`X-E3` → `XE3`, for file name suffixes."""
    exif = read_exif(path)
    model = exif.get(MODEL) if exif is not None else None
    tag = re.sub(r"[^A-Za-z0-9]", "", str(model or "")).upper()
    return tag or None


def collect(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and not is_junk(p.name) and p.suffix.lower() in MEDIA_EXTS
    )


def _suffixes(tag: str | None) -> Iterator[str]:
    yield ""
    if tag:
        yield f"_{tag}"
    n = 2
    while True:
        yield f"_{tag}_{n}" if tag else f"_{n}"
        n += 1


def _fits(dirs: dict[Path, Path], suffix: str) -> bool:
    """No file would land on a different one (same size counts as already sorted)."""
    for p, ext_dir in dirs.items():
        dst = ext_dir / f"{p.stem}{suffix}{p.suffix}"
        if dst.exists() and dst.stat().st_size != p.stat().st_size:
            return False
    return True


def plan(folder: Path) -> list[Move]:
    """Where every loose media file in `folder` should go."""
    files = collect(folder)

    # JPEG dates first, so a RAF+JPG pair always lands on the same day.
    jpeg_dates = {
        p.stem.upper(): capture_date(p) for p in files if p.suffix.lower() in {".jpg", ".jpeg"}
    }
    pairs: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        pairs[p.stem.upper()].append(p)

    moves = []
    for stem, members in pairs.items():
        dirs = {
            p: folder / (jpeg_dates.get(stem) or capture_date(p)) / p.suffix[1:].upper()
            for p in members
        }
        # One suffix for the whole pair, so RAF and JPG keep matching names.
        suffix = ""
        if not _fits(dirs, ""):
            tag = next(filter(None, map(camera_tag, members)), None)
            suffix = next(s for s in _suffixes(tag) if s and _fits(dirs, s))
        for p, ext_dir in dirs.items():
            dst = ext_dir / f"{p.stem}{suffix}{p.suffix}"
            moves.append(Move(p, dst, ext_dir.parent.name, already=dst.exists()))
    return sorted(moves, key=lambda m: m.src)


def ingest(folder: Path, *, dry_run: bool = False) -> None:
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")
    moves = plan(folder)
    if not moves:
        print(f"[piqa] nothing to sort in {folder}")
        return

    done: Counter[tuple[str, str]] = Counter()
    skipped = 0
    for m in moves:
        if m.already:
            print(f"[piqa] skip {m.src.name}: already sorted at {m.dst}")
            skipped += 1
            continue
        if m.dst.name != m.src.name:
            print(f"[piqa] {m.src.name} -> {m.dst.name}: a different file already has that name")
        if not dry_run:
            m.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(m.src, m.dst)
        done[(m.date, m.dst.parent.name)] += 1

    prefix = "[dry run] would " if dry_run else ""
    for date in sorted({d for d, _ in done}):
        counts = ", ".join(f"{ext} {n}" for (d, ext), n in sorted(done.items()) if d == date)
        print(f"{prefix}move -> {folder / date}: {counts}")
    print(
        f"[piqa] {sum(done.values())} files {'to move' if dry_run else 'moved'}"
        f"{f', {skipped} skipped' if skipped else ''}"
    )
