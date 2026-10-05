import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aurora_images.classifier import Prediction
from aurora_images.config import Settings, get_settings
from aurora_images.main import create_app
from aurora_images.storage import LocalImageStorage


class FakeClassifier:
    model_name = "fake-model"

    def classify(self, data: bytes, top_k: int) -> list[Prediction]:
        return [Prediction("aurora", 0.9), Prediction("sky", 0.1)][:top_k]


def make_image(fmt: str = "PNG", size: tuple[int, int] = (32, 24)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 200, 120)).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        upload_dir=tmp_path / "uploads",
        max_upload_bytes=64 * 1024,
        model_path=tmp_path / "missing.onnx",
        labels_path=tmp_path / "missing.txt",
        top_k=2,
    )


def _client(settings: Settings, **kwargs: object) -> Iterator[TestClient]:
    app = create_app(settings, storage=LocalImageStorage(settings.upload_dir), **kwargs)  # type: ignore[arg-type]
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    yield from _client(settings, classifier=FakeClassifier())


@pytest.fixture
def client_without_model(settings: Settings) -> Iterator[TestClient]:
    yield from _client(settings)
