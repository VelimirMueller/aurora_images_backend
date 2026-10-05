"""Image classification behind a small protocol, with an ONNX Runtime implementation."""

import io
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


class Classifier(Protocol):
    model_name: str
    num_classes: int

    def predict(self, data: bytes) -> npt.NDArray[np.float32]:
        """Return one probability per output class, summing to 1."""
        ...


class ModelNotAvailableError(Exception):
    pass


def session_options(threads: int = 0) -> ort.SessionOptions:
    """CPU session options safe for servers that share cores.

    ONNX Runtime busy-spins its worker threads by default. With two processes on the same cores
    (several workers, containers on one host, CPU quotas) the spinning pools starve each other:
    a 7.7 s label embedding did not finish within 10 minutes when two ran at once. Without
    spinning it took 8.3 s alone and 11.6 s each when two ran at once (measured 2026-10-03).
    """
    options = ort.SessionOptions()
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    options.intra_op_num_threads = threads  # 0 = ONNX Runtime's default (physical cores)
    return options


def load_session(path: Path, threads: int = 0) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(path), session_options(threads), providers=["CPUExecutionProvider"]
    )


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
    def __init__(self, model_path: Path, threads: int = 0) -> None:
        if not model_path.is_file():
            raise ModelNotAvailableError(
                f"model missing ({model_path}); run scripts/fetch_model.py"
            )
        self.model_name = model_path.stem
        self._session = load_session(model_path, threads)
        self._input_name: str = self._session.get_inputs()[0].name
        num_classes = self._session.get_outputs()[0].shape[-1]
        if not isinstance(num_classes, int):
            raise ModelNotAvailableError(f"{model_path} has a dynamic output size ({num_classes})")
        self.num_classes = num_classes

    def predict(self, data: bytes) -> npt.NDArray[np.float32]:
        (logits,) = self._session.run(None, {self._input_name: preprocess(data)})
        return softmax(np.asarray(logits, dtype=np.float32)[0])
