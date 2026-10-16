from PIL import Image

from piqa.images import THUMB_TIERS, find_images, has_thumbs, open_image, thumb_path, write_thumbs
from piqa.viewers import build_payload


def test_find_images_skips_hidden_and_cache(tmp_path):
    for rel in [
        "a/DSCF0001.JPG",
        "a/._DSCF0001.JPG",
        ".piqa/thumbs/x.jpg",
        "b/notes.txt",
        "b/scan.png",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    found = [p.relative_to(tmp_path).as_posix() for p in find_images(tmp_path)]
    assert found == ["a/DSCF0001.JPG", "b/scan.png"]


def test_draft_decode_and_thumbnail_tiers(tmp_path):
    src = tmp_path / "big.jpg"
    Image.new("RGB", (6240, 4160), "white").save(src)
    img = open_image(src, min_size=512)
    assert img.size == (780, 520)  # 1/8-scale DCT decode, still >= 512 on the short side

    cache = tmp_path / ".piqa"
    write_thumbs(img, cache, "big.jpg")
    assert has_thumbs(cache, "big.jpg")
    for tier, size in THUMB_TIERS.items():
        with Image.open(thumb_path(cache, "big.jpg", tier)) as thumb:
            assert max(thumb.size) == size


def test_payload_urls_are_relative_and_quoted():
    import numpy as np

    payload = build_payload(
        ["Japon/Prem/été 1.JPG", "Japon/Laura/b.JPG"], np.array([[0.0, 0.0], [2.0, 1.0]]), "top"
    )
    first = payload["items"][0]
    assert first["orig"] == "../Japon/Prem/%C3%A9t%C3%A9%201.JPG"
    assert first["s"] == "thumbs/s/Japon/Prem/%C3%A9t%C3%A9%201.JPG.jpg"
    assert [g["name"] for g in payload["groups"]] == ["Japon"]
    assert payload["items"][1]["x"] == 1.0 and payload["items"][1]["y"] == 0.5
