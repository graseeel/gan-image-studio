from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch

from gan_image_studio.config import ModelConfig
from gan_image_studio.utils import ensure_directory, file_sha256


@dataclass(frozen=True)
class CheckpointMetadata:
    path: Path
    sha256: str
    size_bytes: int


class CheckpointPayload(TypedDict):
    generator: dict[str, Any]
    discriminator: dict[str, Any]
    generator_optimizer: dict[str, Any]
    discriminator_optimizer: dict[str, Any]
    model_config: dict[str, int]
    epoch: int
    step: int
    metrics: dict[str, float]


def save_checkpoint(
    path: Path,
    *,
    generator_state: dict[str, Any],
    discriminator_state: dict[str, Any],
    generator_optimizer_state: dict[str, Any],
    discriminator_optimizer_state: dict[str, Any],
    config: ModelConfig,
    epoch: int,
    step: int,
    metrics: dict[str, float],
) -> CheckpointMetadata:
    ensure_directory(path.parent)
    payload: CheckpointPayload = {
        "generator": generator_state,
        "discriminator": discriminator_state,
        "generator_optimizer": generator_optimizer_state,
        "discriminator_optimizer": discriminator_optimizer_state,
        "model_config": config.to_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    validate_checkpoint(tmp_path)
    os.replace(tmp_path, path)
    return CheckpointMetadata(path=path, sha256=file_sha256(path), size_bytes=path.stat().st_size)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> CheckpointPayload:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return _validate_payload(checkpoint)


def validate_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_payload(checkpoint)


def _validate_payload(payload: object) -> CheckpointPayload:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    required = {
        "generator",
        "discriminator",
        "generator_optimizer",
        "discriminator_optimizer",
        "model_config",
        "epoch",
        "step",
        "metrics",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"checkpoint is missing keys: {', '.join(missing)}")
    return {
        "generator": payload["generator"],
        "discriminator": payload["discriminator"],
        "generator_optimizer": payload["generator_optimizer"],
        "discriminator_optimizer": payload["discriminator_optimizer"],
        "model_config": payload["model_config"],
        "epoch": payload["epoch"],
        "step": payload["step"],
        "metrics": payload["metrics"],
    }
