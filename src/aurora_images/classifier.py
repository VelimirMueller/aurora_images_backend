"""Image classification behind a small protocol, with an ONNX Runtime implementation."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from PIL import Image, ImageOps

_INPUT_SIZE = 224
_RESIZE_SIZE = 256
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


class Classifier(Protocol):
    model_name: str

    def classify(self, data: bytes, top_k: int) -> list[Prediction]: ...


class ModelNotAvailableError(Exception):
    pass


def preprocess(data: bytes) -> npt.NDArray[np.float32]:
    """ImageNet preprocessing: resize the short side to 256, center-crop 224, normalize, NCHW."""
    with Image.open(io.BytesIO(data)) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
    scale = _RESIZE_SIZE / min(rgb.size)
    resized = rgb.resize(
        (max(_INPUT_SIZE, round(rgb.width * scale)), max(_INPUT_SIZE, round(rgb.height * scale))),
        Image.Resampling.BILINEAR,
    )
    left = (resized.width - _INPUT_SIZE) // 2
    top = (resized.height - _INPUT_SIZE) // 2
    cropped = resized.crop((left, top, left + _INPUT_SIZE, top + _INPUT_SIZE))
    array = (np.asarray(cropped, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return array.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    shifted = np.exp(logits - logits.max())
    result: npt.NDArray[np.float32] = shifted / shifted.sum()
    return result


class OnnxClassifier:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        if not model_path.is_file() or not labels_path.is_file():
            raise ModelNotAvailableError(
                f"model files missing ({model_path}, {labels_path}); run scripts/fetch_model.py"
            )
        self.model_name = model_path.stem
        self._labels = labels_path.read_text(encoding="utf-8").splitlines()
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name: str = self._session.get_inputs()[0].name
        num_classes = self._session.get_outputs()[0].shape[-1]
        # Dynamic output dims are symbolic (str/None); only a fixed size can be checked here.
        if isinstance(num_classes, int) and len(self._labels) != num_classes:
            raise ModelNotAvailableError(
                f"{labels_path} has {len(self._labels)} labels, model outputs {num_classes}"
            )

    def classify(self, data: bytes, top_k: int) -> list[Prediction]:
        (logits,) = self._session.run(None, {self._input_name: preprocess(data)})
        probabilities = softmax(np.asarray(logits, dtype=np.float32)[0])
        top = np.argsort(probabilities)[::-1][:top_k]
        return [Prediction(self._labels[i], round(float(probabilities[i]), 4)) for i in top]
