from pathlib import Path

from gan_image_studio.storage import LocalArtifactStore


def test_local_artifact_upload_download(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    source = tmp_path / "source.bin"
    source.write_bytes(b"artifact")

    remote = store.upload_file(source, "generated-images", "user-a/grid.png")
    destination = store.download_file("generated-images", "user-a/grid.png", tmp_path / "out.bin")

    assert remote == "generated-images/user-a/grid.png"
    assert destination.read_bytes() == b"artifact"
