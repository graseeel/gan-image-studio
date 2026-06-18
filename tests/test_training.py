import torch
from torch import nn

from gan_image_studio.config import ModelConfig
from gan_image_studio.models import Discriminator, Generator
from gan_image_studio.training import train_step


def test_train_step_returns_finite_losses() -> None:
    config = ModelConfig(latent_dim=8, generator_features=8, discriminator_features=8)
    generator = Generator(config)
    discriminator = Discriminator(config)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )

    metrics = train_step(
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        nn.BCEWithLogitsLoss(),
        torch.randn(2, 3, 32, 32).clamp(-1, 1),
        latent_dim=config.latent_dim,
        device=torch.device("cpu"),
    )

    assert metrics.generator_loss > 0
    assert metrics.discriminator_loss > 0
