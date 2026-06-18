from __future__ import annotations

from pathlib import Path

import torch

from gan_image_studio.checkpoints import load_checkpoint
from gan_image_studio.config import ModelConfig
from gan_image_studio.models import Generator


def latent_from_seed(seed: int, count: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(count, latent_dim, 1, 1, generator=generator, device=device)


def load_generator_from_checkpoint(path: Path, device: torch.device) -> Generator:
    checkpoint = load_checkpoint(path, map_location=device)
    config = ModelConfig(**checkpoint["model_config"])
    generator = Generator(config).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator


@torch.no_grad()
def generate_images(
    generator: Generator,
    *,
    seed: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("count must be positive")
    latent = latent_from_seed(seed, count, generator.config.latent_dim, device)
    return generator(latent).detach().cpu()


@torch.no_grad()
def interpolate_between_seeds(
    generator: Generator,
    *,
    start_seed: int,
    end_seed: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    start = latent_from_seed(start_seed, 1, generator.config.latent_dim, device)
    end = latent_from_seed(end_seed, 1, generator.config.latent_dim, device)
    weights = torch.linspace(0, 1, steps, device=device).view(steps, 1, 1, 1)
    latents = start * (1 - weights) + end * weights
    return generator(latents).detach().cpu()
