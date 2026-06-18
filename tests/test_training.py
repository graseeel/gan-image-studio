from pathlib import Path

import torch
from PIL import Image
from torch import nn

from gan_image_studio.config import ModelConfig, TrainingConfig
from gan_image_studio.models import Discriminator, Generator
from gan_image_studio.training import train, train_step


def _write_dataset(root: Path, count: int = 4) -> None:
    class_dir = root / "class-a"
    class_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (40, 40), color=(index * 40, 20, 60)).save(
            class_dir / f"image-{index}.png"
        )


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


def test_training_can_resume_from_checkpoint(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    model = ModelConfig(latent_dim=8, generator_features=8, discriminator_features=8)
    first_checkpoint_dir = tmp_path / "checkpoints-a"
    train(
        TrainingConfig(
            dataset="folder",
            data_dir=dataset_dir,
            output_dir=tmp_path / "outputs-a",
            checkpoint_dir=first_checkpoint_dir,
            epochs=1,
            batch_size=2,
            max_batches=1,
            save_every_steps=1,
            sample_every_steps=1,
            num_workers=0,
            model=model,
        )
    )
    checkpoint = next(first_checkpoint_dir.glob("*.pt"))

    resumed_checkpoint_dir = tmp_path / "checkpoints-b"
    train(
        TrainingConfig(
            dataset="folder",
            data_dir=dataset_dir,
            output_dir=tmp_path / "outputs-b",
            checkpoint_dir=resumed_checkpoint_dir,
            epochs=2,
            batch_size=2,
            max_batches=1,
            save_every_steps=1,
            sample_every_steps=1,
            num_workers=0,
            resume_checkpoint=checkpoint,
            model=model,
        )
    )

    assert (resumed_checkpoint_dir / "dcgan-epoch-0001-step-000002.pt").exists()
