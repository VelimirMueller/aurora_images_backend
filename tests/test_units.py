import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aurora_images.classifier import ModelNotAvailableError, OnnxClassifier, preprocess, softmax
from aurora_images.images import InvalidImageError, validate_image
from aurora_images.storage import LocalImageStorage
from tests.conftest import make_image

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL = MODELS_DIR / "mobilenetv2-12.onnx"
LABELS = MODELS_DIR / "imagenet_classes.txt"


@pytest.mark.parametrize("size", [(32, 24), (640, 480), (300, 1000)])
def test_preprocess_produces_normalized_nchw_tensor(size: tuple[int, int]) -> None:
    tensor = preprocess(make_image(size=size))

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32


def test_softmax_is_a_probability_distribution() -> None:
    probabilities = softmax(np.array([1.0, 2.0, 1000.0], dtype=np.float32))

    assert probabilities.sum() == pytest.approx(1.0)
    assert int(probabilities.argmax()) == 2


def test_validate_image_rejects_pixel_bombs() -> None:
    with pytest.raises(InvalidImageError, match="dimensions"):
        validate_image(make_image(size=(200, 200)), max_pixels=10_000)


def test_storage_writes_unique_files(tmp_path: Path) -> None:
    storage = LocalImageStorage(tmp_path)

    first, second = storage.save(b"a", "png"), storage.save(b"b", "png")

    assert first != second
    assert (tmp_path / f"{first}.png").read_bytes() == b"a"


def test_classifier_reports_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError, match="fetch_model"):
        OnnxClassifier(tmp_path / "none.onnx", tmp_path / "none.txt")


needs_model = pytest.mark.skipif(not MODEL.is_file(), reason="run scripts/fetch_model.py first")


@needs_model
def test_classifier_rejects_mismatched_labels(tmp_path: Path) -> None:
    labels = tmp_path / "labels.txt"
    labels.write_text("only\ntwo\n")

    with pytest.raises(ModelNotAvailableError, match="2 labels"):
        OnnxClassifier(MODEL, labels)


def test_storage_removes_partial_file_on_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("aurora_images.storage.os.fsync", fail)

    with pytest.raises(OSError, match="disk full"):
        LocalImageStorage(tmp_path).save(b"a", "png")
    assert list(tmp_path.iterdir()) == []


def test_preprocess_applies_exif_orientation() -> None:
    # Stored landscape: red left half, blue right half. EXIF 6 = display rotated 90 deg CW,
    # so after transposing, red must be on top and blue at the bottom.
    img = Image.new("RGB", (400, 100), (0, 0, 255))
    img.paste((255, 0, 0), (0, 0, 200, 100))
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", exif=exif)

    red = preprocess(buffer.getvalue())[0, 0]

    assert red[:50].mean() > red[-50:].mean()


@needs_model
def test_onnx_classifier_end_to_end() -> None:
    classifier = OnnxClassifier(MODEL, LABELS)

    predictions = classifier.classify(make_image("JPEG", (320, 240)), top_k=5)

    assert len(predictions) == 5
    assert classifier.model_name == "mobilenetv2-12"
    assert all(0 <= p.score <= 1 for p in predictions)
    assert [p.score for p in predictions] == sorted((p.score for p in predictions), reverse=True)
