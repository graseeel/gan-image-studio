from __future__ import annotations

import math

import torch
from torch import nn

from gan_image_studio.config import ModelConfig


class Generator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        upsample_count = int(math.log2(config.image_size // 4))
        start_channels = config.generator_features * min(2**upsample_count, 8)

        layers: list[nn.Module] = [
            nn.ConvTranspose2d(config.latent_dim, start_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(start_channels),
            nn.ReLU(True),
        ]

        current_size = 4
        in_channels = start_channels
        while current_size < config.image_size // 2:
            out_channels = max(config.generator_features, in_channels // 2)
            layers.extend(
                [
                    nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                ]
            )
            in_channels = out_channels
            current_size *= 2

        layers.extend(
            [
                nn.ConvTranspose2d(in_channels, config.image_channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.apply(initialize_dcgan_weights)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 2:
            latent = latent[:, :, None, None]
        return self.net(latent)


class Discriminator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = [
            nn.Conv2d(config.image_channels, config.discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        current_size = config.image_size // 2
        in_channels = config.discriminator_features

        while current_size > 4:
            out_channels = min(in_channels * 2, config.discriminator_features * 8)
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_channels = out_channels
            current_size //= 2

        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 0, bias=False))
        self.net = nn.Sequential(*layers)
        self.apply(initialize_dcgan_weights)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image).view(image.size(0))


def initialize_dcgan_weights(module: nn.Module) -> None:
    # DCGAN uses explicit normal initialization to avoid framework-default drift.
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
