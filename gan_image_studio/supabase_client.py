from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gan_image_studio.checkpoints import CheckpointMetadata
from gan_image_studio.storage import ArtifactStore, SupabaseArtifactStore


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    anon_key: str
    service_role_key: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.anon_key)


class SupabaseGateway:
    def __init__(self, settings: SupabaseSettings) -> None:
        if not settings.enabled:
            raise ValueError("Supabase URL and anon key are required")
        from supabase import create_client

        self.settings = settings
        self.user_client = create_client(settings.url, settings.anon_key)
        self.service_client = (
            create_client(settings.url, settings.service_role_key)
            if settings.service_role_key
            else None
        )
        self.user_artifacts: ArtifactStore = SupabaseArtifactStore(self.user_client)
        self.service_artifacts: ArtifactStore | None = (
            SupabaseArtifactStore(self.service_client) if self.service_client else None
        )

    def sign_in(self, email: str, password: str) -> str:
        response = self.user_client.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        if response.session is None:
            raise PermissionError("Supabase did not return a session")
        return response.session.access_token

    def current_user_id(self) -> str:
        response = self.user_client.auth.get_user()
        if response.user is None:
            raise PermissionError("sign in before accessing user data")
        return str(response.user.id)

    def list_checkpoints(self) -> list[dict[str, Any]]:
        response = (
            self.user_client.table("model_checkpoints")
            .select("id, experiment_id, storage_path, epoch, step, metrics, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        return list(response.data or [])

    def list_generation_history(self) -> list[dict[str, Any]]:
        response = (
            self.user_client.table("generations")
            .select("id, experiment_id, checkpoint_id, storage_path, seed, image_count, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        return list(response.data or [])

    def save_generation_record(
        self,
        *,
        user_id: str,
        storage_path: str,
        seed: int,
        image_count: int,
        checkpoint_id: str | None,
        experiment_id: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "storage_bucket": "generated-images",
            "storage_path": storage_path,
            "seed": seed,
            "image_count": image_count,
            "checkpoint_id": checkpoint_id,
            "experiment_id": experiment_id,
            "metadata": metadata,
        }
        response = self.user_client.table("generations").insert(payload).execute()
        if not response.data:
            raise RuntimeError("generation insert returned no data")
        return dict(response.data[0])

    def favorite_generation(self, generation_id: str) -> None:
        self.user_client.table("generation_favorites").upsert(
            {"user_id": self.current_user_id(), "generation_id": generation_id}
        ).execute()

    def register_checkpoint(
        self,
        checkpoint: CheckpointMetadata,
        *,
        experiment_id: str,
        owner_id: str,
        epoch: int,
        step: int,
        metrics: dict[str, float],
    ) -> dict[str, Any]:
        if self.service_client is None:
            raise PermissionError("service role key is required to register checkpoints")
        if self.service_artifacts is None:
            raise PermissionError("service artifact store is required to register checkpoints")
        remote_path = f"{owner_id}/checkpoints/{checkpoint.path.name}"
        # Checkpoint upload is intentionally backend-only through the service client.
        self.service_artifacts.upload_file(checkpoint.path, "model-checkpoints", remote_path)
        payload = {
            "experiment_id": experiment_id,
            "owner_id": owner_id,
            "storage_bucket": "model-checkpoints",
            "storage_path": remote_path,
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "sha256": checkpoint.sha256,
            "size_bytes": checkpoint.size_bytes,
            "is_validated": True,
        }
        response = self.service_client.table("model_checkpoints").insert(payload).execute()
        if not response.data:
            raise RuntimeError("checkpoint insert returned no data")
        return dict(response.data[0])

    def upload_generated_grid(self, local_path: Path, user_id: str) -> str:
        remote_path = f"{user_id}/generations/{local_path.name}"
        self.user_artifacts.upload_file(local_path, "generated-images", remote_path)
        return remote_path
