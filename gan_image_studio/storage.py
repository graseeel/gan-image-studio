from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from gan_image_studio.utils import ensure_directory


class ArtifactStore(Protocol):
    def upload_file(self, local_path: Path, bucket: str, remote_path: str) -> str:
        ...

    def download_file(self, bucket: str, remote_path: str, local_path: Path) -> Path:
        ...


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def upload_file(self, local_path: Path, bucket: str, remote_path: str) -> str:
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        destination = self.root / bucket / remote_path
        ensure_directory(destination.parent)
        shutil.copy2(local_path, destination)
        return f"{bucket}/{remote_path}"

    def download_file(self, bucket: str, remote_path: str, local_path: Path) -> Path:
        source = self.root / bucket / remote_path
        if not source.is_file():
            raise FileNotFoundError(source)
        ensure_directory(local_path.parent)
        shutil.copy2(source, local_path)
        return local_path


class SupabaseArtifactStore:
    def __init__(self, client: object) -> None:
        self.client = client

    def upload_file(self, local_path: Path, bucket: str, remote_path: str) -> str:
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        with local_path.open("rb") as handle:
            self.client.storage.from_(bucket).upload(remote_path, handle)
        return f"{bucket}/{remote_path}"

    def download_file(self, bucket: str, remote_path: str, local_path: Path) -> Path:
        data = self.client.storage.from_(bucket).download(remote_path)
        ensure_directory(local_path.parent)
        local_path.write_bytes(data)
        return local_path
