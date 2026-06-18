from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import datasets, transforms

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ImageValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ImageIssue:
    path: Path
    reason: str


@dataclass(frozen=True)
class DatasetInspection:
    root: Path
    valid_count: int
    issues: tuple[ImageIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues and self.valid_count > 0


def normalized_image_transform(image_size: int) -> transforms.Compose:
    # The Tanh generator emits [-1, 1], so training inputs use the same interval.
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _inspect_image(path: Path, image_size: int) -> ImageIssue | None:
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return ImageIssue(path, f"unsupported extension {path.suffix}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError) as exc:
        return ImageIssue(path, f"corrupt or unreadable image: {exc}")

    if width < image_size or height < image_size:
        return ImageIssue(path, f"image is smaller than requested size {image_size}")
    return None


def inspect_image_folder(root: Path, image_size: int) -> DatasetInspection:
    if not root.exists():
        raise ImageValidationError(f"dataset folder does not exist: {root}")
    if not root.is_dir():
        raise ImageValidationError(f"dataset path is not a directory: {root}")

    valid_count = 0
    issues: list[ImageIssue] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        issue = _inspect_image(path, image_size)
        if issue is None:
            valid_count += 1
        else:
            issues.append(issue)

    if valid_count == 0 and not issues:
        issues.append(ImageIssue(root, "no image files found"))
    return DatasetInspection(root=root, valid_count=valid_count, issues=tuple(issues))


class ValidatedImageFolder(Dataset):
    def __init__(self, root: Path, image_size: int) -> None:
        inspection = inspect_image_folder(root, image_size)
        if not inspection.ok:
            joined = "; ".join(f"{issue.path}: {issue.reason}" for issue in inspection.issues)
            raise ImageValidationError(joined)
        self.dataset = datasets.ImageFolder(
            root=str(root),
            transform=normalized_image_transform(image_size),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, _class_id = self.dataset[index]
        return image


def build_training_dataset(
    kind: str,
    data_dir: Path,
    image_size: int,
    download: bool = True,
) -> Dataset:
    if kind == "cifar10":
        return datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=download,
            transform=normalized_image_transform(image_size),
        )
    if kind == "folder":
        return ValidatedImageFolder(data_dir, image_size)
    raise ImageValidationError(f"unsupported dataset kind: {kind}")
