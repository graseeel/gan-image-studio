from __future__ import annotations

from pathlib import Path

import pytest

from gan_image_studio import app as app_module
from gan_image_studio.supabase_client import AuthSession


class GatewayStub:
    def __init__(self) -> None:
        self.saved: list[tuple[str, int, int]] = []
        self.favorites: list[str] = []

    def sign_in(self, email: str, password: str) -> AuthSession:
        assert email == "user@example.com"
        assert password == "secret"
        return AuthSession(access_token="access", refresh_token="refresh", user_id="user-a")

    def current_user_id(self) -> str:
        return "user-a"

    def upload_generated_grid(self, local_path: Path, user_id: str) -> str:
        assert user_id == "user-a"
        return f"{user_id}/generations/{local_path.name}"

    def save_generation_record(
        self,
        *,
        user_id: str,
        storage_path: str,
        seed: int,
        image_count: int,
        checkpoint_id: str | None,
        experiment_id: str | None,
        metadata: dict[str, object],
    ) -> dict[str, str]:
        self.saved.append((storage_path, seed, image_count))
        assert user_id == "user-a"
        assert checkpoint_id is None
        assert experiment_id is None
        assert metadata == {"source": "gradio"}
        return {"id": "generation-a"}

    def favorite_generation(self, generation_id: str) -> None:
        self.favorites.append(generation_id)

    def list_generation_history(self) -> list[dict[str, object]]:
        return [
            {
                "id": "generation-a",
                "seed": 7,
                "image_count": 4,
                "storage_path": "user-a/generations/grid.png",
                "created_at": "2026-06-18T00:00:00Z",
            }
        ]


def test_sign_in_returns_hidden_session_state(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = GatewayStub()
    monkeypatch.setattr(app_module, "_gateway", lambda session=None: gateway)

    status, session = app_module._sign_in("user@example.com", "secret")

    assert status == "signed in"
    assert session == AuthSession(access_token="access", refresh_token="refresh", user_id="user-a")


def test_authenticated_ui_actions_reuse_session(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = GatewayStub()
    sessions_seen: list[AuthSession | None] = []

    def gateway_factory(session: AuthSession | None = None) -> GatewayStub:
        sessions_seen.append(session)
        return gateway

    session = AuthSession(access_token="access", refresh_token="refresh", user_id="user-a")
    monkeypatch.setattr(app_module, "_gateway", gateway_factory)

    saved_id = app_module._save_generation("/tmp/grid.png", seed=7, count=4, session=session)
    rows = app_module._history(session)
    favorite_status = app_module._favorite_generation("generation-a", session)

    assert saved_id == "generation-a"
    assert rows == [
        [
            "generation-a",
            "7",
            "4",
            "user-a/generations/grid.png",
            "2026-06-18T00:00:00Z",
        ]
    ]
    assert favorite_status == "favorited"
    assert gateway.saved == [("user-a/generations/grid.png", 7, 4)]
    assert gateway.favorites == ["generation-a"]
    assert sessions_seen == [session, session, session]
