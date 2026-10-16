import os
import time
from pathlib import Path

from PIL import Image

from piqa.ingest import capture_date, ingest, plan


def make_jpeg(
    path: Path, taken: str | None = None, model: str | None = None, width: int = 64
) -> Path:
    """Different widths give different file sizes, i.e. "a different file" for ingest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, 48), "gray")
    exif = Image.Exif()
    if taken:
        exif.get_ifd(0x8769)[36867] = taken
    if model:
        exif[0x0110] = model
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
    moves = {m.src.name: m.dst for m in plan(tmp_path)}
    assert moves["DSCF0003.JPG"] == tmp_path / "2026-08-22" / "JPG" / "DSCF0003.JPG"
    assert moves["DSCF0003.RAF"] == tmp_path / "2026-08-22" / "RAF" / "DSCF0003.RAF"


def test_sorting_import_in_place_only_touches_loose_files(tmp_path):
    make_jpeg(tmp_path / "DSCF0004.JPG", "2025:02:15 09:00:00")
    make_jpeg(tmp_path / "2025-02-14" / "JPG" / "DSCF0001.JPG", "2025:02:14 09:00:00")
    (tmp_path / "._DSCF0004.JPG").write_bytes(b"appledouble")
    ingest(tmp_path)
    assert (tmp_path / "2025-02-15" / "JPG" / "DSCF0004.JPG").exists()
    assert not (tmp_path / "DSCF0004.JPG").exists()
    assert (tmp_path / "._DSCF0004.JPG").exists()  # junk left alone
    assert (tmp_path / "2025-02-14" / "JPG" / "DSCF0001.JPG").exists()


def test_name_clash_gets_camera_suffix_for_the_whole_pair(tmp_path):
    day = tmp_path / "2025-02-14"
    make_jpeg(day / "JPG" / "DSCF0412.JPG", "2025:02:14 09:00:00", "X-T3")
    touch(day / "RAF" / "DSCF0412.RAF", "2025-02-14")
    make_jpeg(tmp_path / "DSCF0412.JPG", "2025:02:14 18:00:00", "X-E3", width=80)
    (tmp_path / "DSCF0412.RAF").write_bytes(b"other body raw, different size")
    ingest(tmp_path)
    assert (day / "JPG" / "DSCF0412_XE3.JPG").exists()
    assert (day / "RAF" / "DSCF0412_XE3.RAF").exists()
    assert not (tmp_path / "DSCF0412.JPG").exists()
    assert not (tmp_path / "DSCF0412.RAF").exists()


def test_clash_without_model_or_with_taken_suffix_counts_up(tmp_path):
    day = tmp_path / "2025-02-14" / "JPG"
    make_jpeg(day / "DSCF0001.JPG", "2025:02:14 09:00:00")
    make_jpeg(tmp_path / "DSCF0001.JPG", "2025:02:14 10:00:00", width=80)
    make_jpeg(day / "DSCF0002.JPG", "2025:02:14 09:00:00", "X-E3")
    make_jpeg(day / "DSCF0002_XE3.JPG", "2025:02:14 09:30:00", "X-E3", width=96)
    make_jpeg(tmp_path / "DSCF0002.JPG", "2025:02:14 11:00:00", "X-E3", width=80)
    dsts = {m.src.name: m.dst.name for m in plan(tmp_path)}
    assert dsts == {"DSCF0001.JPG": "DSCF0001_2.JPG", "DSCF0002.JPG": "DSCF0002_XE3_2.JPG"}


def test_already_sorted_file_is_skipped_and_left_in_place(tmp_path):
    make_jpeg(tmp_path / "2025-02-14" / "JPG" / "DSCF0007.JPG", "2025:02:14 09:00:00")
    make_jpeg(tmp_path / "DSCF0007.JPG", "2025:02:14 09:00:00")
    ingest(tmp_path)
    assert (tmp_path / "DSCF0007.JPG").exists()
    assert not (tmp_path / "2025-02-14" / "JPG" / "DSCF0007_2.JPG").exists()


def test_dry_run_changes_nothing(tmp_path):
    make_jpeg(tmp_path / "DSCF0006.JPG", "2026:05:02 12:00:00")
    ingest(tmp_path, dry_run=True)
    assert (tmp_path / "DSCF0006.JPG").exists()
    assert not (tmp_path / "2026-05-02").exists()
