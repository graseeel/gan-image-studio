from pathlib import Path

import torch

from gan_image_studio.checkpoints import load_checkpoint, save_checkpoint
from gan_image_studio.config import ModelConfig
from gan_image_studio.models import Discriminator, Generator


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    config = ModelConfig(latent_dim=8, generator_features=8, discriminator_features=8)
    generator = Generator(config)
    discriminator = Discriminator(config)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    metadata = save_checkpoint(
        tmp_path / "model.pt",
        generator_state=generator.state_dict(),
        discriminator_state=discriminator.state_dict(),
        generator_optimizer_state=generator_optimizer.state_dict(),
        discriminator_optimizer_state=discriminator_optimizer.state_dict(),
        config=config,
        epoch=1,
        step=2,
        metrics={"generator_loss": 1.0, "discriminator_loss": 2.0},
    )
    checkpoint = load_checkpoint(metadata.path)

    assert metadata.path.exists()
    assert metadata.size_bytes > 0
    assert len(metadata.sha256) == 64
    assert checkpoint["epoch"] == 1
    assert checkpoint["step"] == 2
