from pathlib import Path

from gan_image_studio.supabase_client import SupabaseTrainingRegistry


class _GatewayStub:
    def __init__(self) -> None:
        self.sample_calls: list[tuple[Path, str, str, int, int]] = []
        self.checkpoint_calls: list[tuple[object, str, str, int, int, dict[str, float]]] = []

    def register_sample_grid(
        self,
        path: Path,
        *,
        experiment_id: str,
        owner_id: str,
        epoch: int,
        step: int,
    ) -> None:
        self.sample_calls.append((path, experiment_id, owner_id, epoch, step))

    def register_checkpoint(
        self,
        checkpoint: object,
        *,
        experiment_id: str,
        owner_id: str,
        epoch: int,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        self.checkpoint_calls.append((checkpoint, experiment_id, owner_id, epoch, step, metrics))


def test_training_registry_binds_supabase_experiment_context(tmp_path: Path) -> None:
    gateway = _GatewayStub()
    registry = SupabaseTrainingRegistry(
        gateway=gateway,
        experiment_id="experiment-a",
        owner_id="user-a",
    )

    sample = tmp_path / "sample.png"
    registry.register_sample_grid(sample, epoch=1, step=2)
    registry.register_checkpoint(object(), epoch=3, step=4, metrics={"generator_loss": 1.0})

    assert gateway.sample_calls == [(sample, "experiment-a", "user-a", 1, 2)]
    assert gateway.checkpoint_calls[0][1:] == (
        "experiment-a",
        "user-a",
        3,
        4,
        {"generator_loss": 1.0},
    )
