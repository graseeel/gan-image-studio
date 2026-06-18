from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 100
    image_channels: int = 3
    image_size: int = 32
    generator_features: int = 64
    discriminator_features: int = 64

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.image_channels not in {1, 3}:
            raise ValueError("image_channels must be 1 or 3")
        if self.image_size < 32 or not _is_power_of_two(self.image_size):
            raise ValueError("image_size must be a power of two and at least 32")
        if self.generator_features <= 0 or self.discriminator_features <= 0:
            raise ValueError("feature counts must be positive")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    dataset: str = "cifar10"
    data_dir: Path = Path("datasets")
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("checkpoints")
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    seed: int = 42
    save_every_steps: int = 500
    sample_every_steps: int = 250
    num_workers: int = 2
    max_batches: int | None = None
    device: str = "cpu"
    resume_checkpoint: Path | None = None
    model: ModelConfig = ModelConfig()

    def __post_init__(self) -> None:
        if self.dataset not in {"cifar10", "folder"}:
            raise ValueError("dataset must be 'cifar10' or 'folder'")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("optimizer betas must be in [0, 1)")
        if self.save_every_steps <= 0 or self.sample_every_steps <= 0:
            raise ValueError("save/sample intervals must be positive")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("max_batches must be positive when provided")

    @classmethod
    def quick_cpu(cls) -> TrainingConfig:
        return cls(
            epochs=1,
            batch_size=8,
            num_workers=0,
            max_batches=2,
            save_every_steps=2,
            sample_every_steps=1,
            model=ModelConfig(generator_features=16, discriminator_features=16),
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["data_dir"] = str(self.data_dir)
        data["output_dir"] = str(self.output_dir)
        data["checkpoint_dir"] = str(self.checkpoint_dir)
        data["resume_checkpoint"] = str(self.resume_checkpoint) if self.resume_checkpoint else None
        return data
