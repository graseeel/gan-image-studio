import torch

from gan_image_studio.config import ModelConfig
from gan_image_studio.models import Discriminator, Generator


def test_generator_dimensions_and_output_range() -> None:
    config = ModelConfig(latent_dim=16, generator_features=8, discriminator_features=8)
    generator = Generator(config)
    latent = torch.randn(4, config.latent_dim)

    images = generator(latent)

    assert images.shape == (4, 3, 32, 32)
    assert float(images.detach().min()) >= -1.0
    assert float(images.detach().max()) <= 1.0


def test_discriminator_dimensions() -> None:
    config = ModelConfig(latent_dim=16, generator_features=8, discriminator_features=8)
    discriminator = Discriminator(config)

    logits = discriminator(torch.randn(4, 3, 32, 32))

    assert logits.shape == (4,)
