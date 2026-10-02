from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aurora_images.config import Settings
from tests.conftest import make_image


def test_health_reports_model_state(client: TestClient, client_without_model: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert client_without_model.get("/health").json() == {"status": "ok", "model_loaded": False}


def test_upload_stores_image_under_generated_id(client: TestClient, settings: Settings) -> None:
    response = client.post("/v1/images", files={"image": ("photo.png", make_image(), "image/png")})

    assert response.status_code == 201
    body = response.json()
    assert body["content_type"] == "image/png"
    assert (body["width"], body["height"]) == (32, 24)
    assert (settings.upload_dir / f"{body['id']}.png").is_file()


@pytest.mark.parametrize(
    "filename", ["../../evil.png", "/etc/passwd", "..\\..\\evil.png", "uploads/../../x.py"]
)
def test_client_filename_never_reaches_the_filesystem(
    client: TestClient, settings: Settings, tmp_path: Path, filename: str
) -> None:
    response = client.post("/v1/images", files={"image": (filename, make_image(), "image/png")})

    assert response.status_code == 201
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written == [settings.upload_dir / f"{response.json()['id']}.png"]


def test_extension_comes_from_decoded_format_not_from_client(
    client: TestClient, settings: Settings
) -> None:
    jpeg = make_image("JPEG")
    response = client.post("/v1/images", files={"image": ("shell.php", jpeg, "image/png")})

    assert response.status_code == 201
    assert response.json()["content_type"] == "image/jpeg"
    assert (settings.upload_dir / f"{response.json()['id']}.jpg").is_file()


def test_non_image_is_rejected(client: TestClient, settings: Settings) -> None:
    payload = b"<?php system($_GET['c']); ?>"
    response = client.post("/v1/images", files={"image": ("a.png", payload, "image/png")})

    assert response.status_code == 415
    assert not settings.upload_dir.exists()


def test_unsupported_format_is_rejected(client: TestClient) -> None:
    gif = make_image("GIF")
    response = client.post("/v1/images", files={"image": ("a.gif", gif, "image/gif")})

    assert response.status_code == 415
    assert "unsupported image format" in response.json()["detail"]


def test_empty_upload_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/images", files={"image": ("a.png", b"", "image/png")})

    assert response.status_code == 415


def test_oversized_upload_is_rejected(client: TestClient, settings: Settings) -> None:
    payload = b"\x89PNG" + b"\0" * settings.max_upload_bytes
    response = client.post("/v1/images", files={"image": ("big.png", payload, "image/png")})

    assert response.status_code == 413
    assert not settings.upload_dir.exists()


def test_classification_returns_top_k(client: TestClient, settings: Settings) -> None:
    response = client.post(
        "/v1/classifications", files={"image": ("a.webp", make_image("WEBP"), "image/webp")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "fake-model",
        "predictions": [{"label": "aurora", "score": 0.9}, {"label": "sky", "score": 0.1}],
    }
    assert not settings.upload_dir.exists(), "classification must not persist the image"


def test_classification_without_model_is_503(client_without_model: TestClient) -> None:
    response = client_without_model.post(
        "/v1/classifications", files={"image": ("a.png", make_image(), "image/png")}
    )

    assert response.status_code == 503


def test_cors_allows_only_configured_origins(client: TestClient) -> None:
    allowed = client.get("/health", headers={"Origin": "http://localhost:5173"})
    denied = client.get("/health", headers={"Origin": "https://evil.example"})

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in denied.headers
    assert "access-control-allow-credentials" not in allowed.headers
