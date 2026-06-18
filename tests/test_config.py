import pytest

from gan_image_studio.config import ModelConfig, TrainingConfig


def test_invalid_model_config_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(latent_dim=0)

    with pytest.raises(ValueError):
        ModelConfig(image_size=48)


def test_invalid_training_config_rejected() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0)
