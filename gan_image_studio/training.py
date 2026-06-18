from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gan_image_studio.checkpoints import CheckpointMetadata, load_checkpoint, save_checkpoint
from gan_image_studio.config import TrainingConfig
from gan_image_studio.data import build_training_dataset
from gan_image_studio.models import Discriminator, Generator
from gan_image_studio.plotting import save_image_grid
from gan_image_studio.utils import ensure_directory, set_reproducible_seed, write_json


class ExperimentRegistry(Protocol):
    def register_checkpoint(
        self,
        checkpoint: CheckpointMetadata,
        *,
        epoch: int,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        ...

    def register_sample_grid(self, path: Path, *, epoch: int, step: int) -> None:
        ...


@dataclass(frozen=True)
class StepMetrics:
    discriminator_loss: float
    generator_loss: float


def has_invalid_gradients(module: nn.Module) -> bool:
    for parameter in module.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return True
    return False


def train_step(
    generator: Generator,
    discriminator: Discriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    real_images: torch.Tensor,
    *,
    latent_dim: int,
    device: torch.device,
) -> StepMetrics:
    batch_size = real_images.size(0)
    real_images = real_images.to(device)
    real_targets = torch.ones(batch_size, device=device)
    fake_targets = torch.zeros(batch_size, device=device)

    # Update D first. Fake images are detached so this optimizer cannot change G.
    discriminator_optimizer.zero_grad(set_to_none=True)
    real_logits = discriminator(real_images)
    discriminator_real_loss = criterion(real_logits, real_targets)
    latent = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    fake_images = generator(latent)
    fake_logits = discriminator(fake_images.detach())
    discriminator_fake_loss = criterion(fake_logits, fake_targets)
    discriminator_loss = discriminator_real_loss + discriminator_fake_loss
    _assert_finite(discriminator_loss, "discriminator_loss")
    discriminator_loss.backward()
    if has_invalid_gradients(discriminator):
        raise FloatingPointError("invalid discriminator gradients")
    discriminator_optimizer.step()

    # Update G second against fresh discriminator scores after D has learned this batch.
    generator_optimizer.zero_grad(set_to_none=True)
    generated_logits = discriminator(fake_images)
    generator_loss = criterion(generated_logits, real_targets)
    _assert_finite(generator_loss, "generator_loss")
    generator_loss.backward()
    if has_invalid_gradients(generator):
        raise FloatingPointError("invalid generator gradients")
    generator_optimizer.step()

    return StepMetrics(
        discriminator_loss=float(discriminator_loss.detach().cpu()),
        generator_loss=float(generator_loss.detach().cpu()),
    )


def train(config: TrainingConfig, registry: ExperimentRegistry | None = None) -> None:
    set_reproducible_seed(config.seed)
    device = torch.device(
        config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
    )
    output_dir = ensure_directory(config.output_dir)
    checkpoint_dir = ensure_directory(config.checkpoint_dir)
    sample_dir = ensure_directory(output_dir / "samples")
    write_json(output_dir / "training-config.json", config.to_dict())

    dataset = build_training_dataset(
        config.dataset,
        config.data_dir,
        config.model.image_size,
        download=config.dataset == "cifar10",
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    generator = Generator(config.model).to(device)
    discriminator = Discriminator(config.model).to(device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2)
    )
    criterion = nn.BCEWithLogitsLoss()
    start_epoch = 0
    step = 0
    if config.resume_checkpoint is not None:
        checkpoint = load_checkpoint(config.resume_checkpoint, map_location=device)
        if checkpoint["model_config"] != config.model.to_dict():
            raise ValueError("resume checkpoint model_config does not match the training config")
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        step = int(checkpoint["step"])

    fixed_generator = torch.Generator(device=device).manual_seed(config.seed)
    fixed_noise = torch.randn(
        64,
        config.model.latent_dim,
        1,
        1,
        generator=fixed_generator,
        device=device,
    )

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    try:
        for epoch in range(start_epoch, config.epochs):
            for batch_index, batch in enumerate(loader):
                real_images = batch[0] if isinstance(batch, (tuple, list)) else batch
                metrics = train_step(
                    generator,
                    discriminator,
                    generator_optimizer,
                    discriminator_optimizer,
                    criterion,
                    real_images,
                    latent_dim=config.model.latent_dim,
                    device=device,
                )
                step += 1
                writer.add_scalar("loss/discriminator", metrics.discriminator_loss, step)
                writer.add_scalar("loss/generator", metrics.generator_loss, step)

                if step % config.sample_every_steps == 0:
                    with torch.no_grad():
                        sample_images = generator(fixed_noise)
                    sample_path = sample_dir / f"epoch-{epoch:04d}-step-{step:06d}.png"
                    save_image_grid(sample_images, sample_path)
                    if registry is not None:
                        registry.register_sample_grid(sample_path, epoch=epoch, step=step)

                if step % config.save_every_steps == 0:
                    _save_and_register_checkpoint(
                        checkpoint_dir,
                        generator,
                        discriminator,
                        generator_optimizer,
                        discriminator_optimizer,
                        config,
                        epoch,
                        step,
                        metrics,
                        registry,
                    )

                if config.max_batches is not None and batch_index + 1 >= config.max_batches:
                    break
    finally:
        writer.close()


def _save_and_register_checkpoint(
    checkpoint_dir: Path,
    generator: Generator,
    discriminator: Discriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    step: int,
    metrics: StepMetrics,
    registry: ExperimentRegistry | None,
) -> CheckpointMetadata:
    metric_payload = {
        "generator_loss": metrics.generator_loss,
        "discriminator_loss": metrics.discriminator_loss,
    }
    checkpoint = save_checkpoint(
        checkpoint_dir / f"dcgan-epoch-{epoch:04d}-step-{step:06d}.pt",
        generator_state=generator.state_dict(),
        discriminator_state=discriminator.state_dict(),
        generator_optimizer_state=generator_optimizer.state_dict(),
        discriminator_optimizer_state=discriminator_optimizer.state_dict(),
        config=config.model,
        epoch=epoch,
        step=step,
        metrics=metric_payload,
    )
    # Remote registration only happens after the local artifact has been validated and hashed.
    if registry is not None:
        registry.register_checkpoint(checkpoint, epoch=epoch, step=step, metrics=metric_payload)
    return checkpoint


def _assert_finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} is not finite")
