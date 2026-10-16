import os
import time
from pathlib import Path

from PIL import Image

from piqa.ingest import capture_date, ingest, plan


def make_jpeg(path: Path, taken: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 48), "gray")
    exif = Image.Exif()
    if taken:
        exif.get_ifd(0x8769)[36867] = taken
    img.save(path, exif=exif)
    return path


def touch(path: Path, mtime: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")
    ts = time.mktime(time.strptime(mtime, "%Y-%m-%d"))
    os.utime(path, (ts, ts))
    return path


def test_capture_date_prefers_exif_over_mtime(tmp_path):
    jpg = make_jpeg(tmp_path / "DSCF0001.JPG", "2025:02:14 18:30:00")
    os.utime(jpg, (0, 0))
    assert capture_date(jpg) == "2025-02-14"


def test_capture_date_falls_back_to_mtime(tmp_path):
    assert capture_date(touch(tmp_path / "DSCF0002.MOV", "2024-03-23")) == "2024-03-23"


def test_raf_follows_its_jpeg(tmp_path):
    make_jpeg(tmp_path / "DSCF0003.JPG", "2026:08:22 10:00:00")
    touch(tmp_path / "DSCF0003.RAF", "2020-01-01")
    moves = {m.src.name: m.dst for m in plan(tmp_path, tmp_path)}
    assert moves["DSCF0003.JPG"] == tmp_path / "2026-08-22" / "JPG" / "DSCF0003.JPG"
    assert moves["DSCF0003.RAF"] == tmp_path / "2026-08-22" / "RAF" / "DSCF0003.RAF"


def test_sorting_import_in_place_only_touches_loose_files(tmp_path):
    make_jpeg(tmp_path / "DSCF0004.JPG", "2025:02:15 09:00:00")
    make_jpeg(tmp_path / "2025-02-14" / "JPG" / "DSCF0001.JPG", "2025:02:14 09:00:00")
    (tmp_path / "._DSCF0004.JPG").write_bytes(b"appledouble")
    ingest(tmp_path, tmp_path, copy=False)
    assert (tmp_path / "2025-02-15" / "JPG" / "DSCF0004.JPG").exists()
    assert not (tmp_path / "DSCF0004.JPG").exists()
    assert (tmp_path / "._DSCF0004.JPG").exists()  # junk left alone
    assert (tmp_path / "2025-02-14" / "JPG" / "DSCF0001.JPG").exists()


def test_card_import_copies_recursively_and_skips_existing(tmp_path):
    card = tmp_path / "card" / "DCIM" / "100_FUJI"
    dest = tmp_path / "Import"
    make_jpeg(card / "DSCF0005.JPG", "2026:05:01 12:00:00")
    ingest(tmp_path / "card", dest, copy=True)
    assert (card / "DSCF0005.JPG").exists()
    assert (dest / "2026-05-01" / "JPG" / "DSCF0005.JPG").exists()
    ingest(tmp_path / "card", dest, copy=True)  # second run: already imported, no error


def test_dry_run_changes_nothing(tmp_path):
    make_jpeg(tmp_path / "DSCF0006.JPG", "2026:05:02 12:00:00")
    ingest(tmp_path, tmp_path, copy=False, dry_run=True)
    assert (tmp_path / "DSCF0006.JPG").exists()
    assert not (tmp_path / "2026-05-02").exists()
