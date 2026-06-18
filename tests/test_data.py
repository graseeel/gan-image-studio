from pathlib import Path

import pytest
from PIL import Image

from gan_image_studio.data import ImageValidationError, ValidatedImageFolder, inspect_image_folder


def _write_image(path: Path, size: tuple[int, int] = (40, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 40, 60)).save(path)


def test_inspect_image_folder_reports_valid_images(tmp_path: Path) -> None:
    _write_image(tmp_path / "class-a" / "sample.png")

    inspection = inspect_image_folder(tmp_path, image_size=32)

    assert inspection.ok
    assert inspection.valid_count == 1
    assert inspection.issues == ()


def test_inspect_image_folder_reports_invalid_files(tmp_path: Path) -> None:
    _write_image(tmp_path / "class-a" / "small.png", size=(16, 16))
    (tmp_path / "class-a" / "bad.txt").write_text("not an image", encoding="utf-8")

    inspection = inspect_image_folder(tmp_path, image_size=32)

    assert not inspection.ok
    assert {issue.reason.split()[0] for issue in inspection.issues} == {
        "image",
        "unsupported",
    }


def test_validated_image_folder_rejects_corrupt_image(tmp_path: Path) -> None:
    corrupt = tmp_path / "class-a" / "bad.png"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"nope")

    with pytest.raises(ImageValidationError):
        ValidatedImageFolder(tmp_path, image_size=32)
