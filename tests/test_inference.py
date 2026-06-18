import torch

from gan_image_studio.config import ModelConfig
from gan_image_studio.inference import generate_images
from gan_image_studio.models import Generator


def test_generation_is_deterministic_for_seed() -> None:
    config = ModelConfig(latent_dim=8, generator_features=8, discriminator_features=8)
    generator = Generator(config)
    device = torch.device("cpu")

    first = generate_images(generator, seed=123, count=2, device=device)
    second = generate_images(generator, seed=123, count=2, device=device)

    assert torch.equal(first, second)
