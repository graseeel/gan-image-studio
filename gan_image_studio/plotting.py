from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from gan_image_studio.utils import ensure_directory


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return (images.clamp(-1, 1) + 1.0) / 2.0


def save_image_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> Path:
    ensure_directory(path.parent)
    grid = make_grid(denormalize(images.detach().cpu()), nrow=nrow, padding=2)
    save_image(grid, path)
    return path
