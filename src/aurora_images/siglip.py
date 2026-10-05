"""Open-vocabulary classification with SigLIP 2: score an image against text labels.

The image and every label are embedded into the same 768-d space. A label's probability is
softmax(exp(logit_scale) * cosine) over all labels, so any text label works without training.
Label embeddings depend only on the text, so they are computed once and cached on disk; the
text tower is loaded only on a cache miss.
"""

import hashlib
import io
import logging
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps
from tokenizers import Tokenizer

from aurora_images.classifier import ModelNotAvailableError, load_session, softmax
from aurora_images.taxonomy import Taxonomy, TaxonomyError

logger = logging.getLogger("aurora_images")

MODEL_NAME = "siglip2-base-patch16-224"
# onnx-community/siglip2-base-patch16-224-ONNX revision, pinned in scripts/fetch_model.py.
REVISION = "ba1f3b0843f24bc5417d38e19c37b287d719b2f4"
VISION_FILE = "vision_model.onnx"
# fp32, not int8: the dynamically quantized text tower gives architecture-dependent embeddings
# (cosine to fp32 0.996 on ARM but 0.959 mean / 0.872 min on x86 without VNNI), which flattened
# the label distribution and broke the eval on CI. fp32 is identical on every architecture.
TEXT_FILE = "text_model.onnx"
TOKENIZER_FILE = "tokenizer.json"
# Learned temperature of google/siglip2-base-patch16-224 (exp(logit_scale), logit_scale=4.7245).
LOGIT_SCALE = float(np.exp(4.724453449249268))
PROMPT_TEMPLATE = "this is a photo of a {}."
_IMAGE_SIZE = 224
# The text tower pools the LAST position, so every prompt is padded to exactly this length.
_TEXT_LENGTH = 64
_TEXT_BATCH = 32


class Session(Protocol):
    def run(
        self, output_names: list[str] | None, input_feed: dict[str, npt.NDArray[np.generic]]
    ) -> list[npt.NDArray[np.float32]]: ...


def preprocess(data: bytes) -> npt.NDArray[np.float32]:
    """SigLIP preprocessing: resize to 224x224 (no crop), scale to [-1, 1], NCHW."""
    with Image.open(io.BytesIO(data)) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
    resized = rgb.resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    return array.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def normalize(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    result: npt.NDArray[np.float32] = (vectors / np.maximum(norms, 1e-12)).astype(np.float32)
    return result


class TextEncoder:
    """Tokenize prompts and embed them with the text tower, loaded on first use."""

    def __init__(self, tokenizer: Tokenizer, session_factory: "SessionFactory") -> None:
        self._tokenizer = tokenizer
        self._tokenizer.enable_truncation(_TEXT_LENGTH)
        self._pad_id = _token_id(tokenizer, "<pad>")
        self._eos_id = _token_id(tokenizer, "<eos>")
        self._tokenizer.enable_padding(length=_TEXT_LENGTH, pad_id=self._pad_id, pad_token="<pad>")  # noqa: S106
        self._session_factory = session_factory
        self._session: Session | None = None

    def token_ids(self, texts: Sequence[str]) -> npt.NDArray[np.int64]:
        rows = []
        for encoding in self._tokenizer.encode_batch([text.lower() for text in texts]):
            ids = list(encoding.ids)
            # Truncation can cut the EOS the tokenizer appends; the model expects it.
            if self._eos_id not in ids:
                ids[-1] = self._eos_id
            rows.append(ids)
        return np.array(rows, dtype=np.int64)

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        if self._session is None:
            self._session = self._session_factory()
        chunks = []
        for start in range(0, len(texts), _TEXT_BATCH):
            ids = self.token_ids(texts[start : start + _TEXT_BATCH])
            (pooled,) = self._session.run(["pooler_output"], {"input_ids": ids})
            chunks.append(np.asarray(pooled, dtype=np.float32))
        return normalize(np.concatenate(chunks))

    def close(self) -> None:
        """Drop the text tower (~3 GB peak while embedding) once the label embeddings exist."""
        self._session = None


class SessionFactory(Protocol):
    def __call__(self) -> Session: ...


class LabelEmbeddingCache:
    """Label embeddings on disk, keyed by model revision, prompt template and label texts."""

    def __init__(self, directory: Path, fingerprint: str = "") -> None:
        self._directory = directory
        self._fingerprint = fingerprint  # identifies the tokenizer + text tower files

    def key(self, texts: Sequence[str]) -> str:
        digest = hashlib.sha256()
        for part in (MODEL_NAME, REVISION, self._fingerprint, PROMPT_TEMPLATE, *texts):
            digest.update(part.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()[:32]

    def load(
        self, texts: Sequence[str], width: int | None = None
    ) -> npt.NDArray[np.float32] | None:
        path = self._directory / f"{self.key(texts)}.npy"
        if not path.is_file():
            return None
        try:
            embeddings = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            logger.warning("ignoring unreadable label cache %s", path)
            return None
        expected = (len(texts), width if width is not None else embeddings.shape[-1])
        if embeddings.shape != expected:
            logger.warning(
                "ignoring label cache %s: shape %s, expected %s", path, embeddings.shape, expected
            )
            return None
        return np.asarray(embeddings, dtype=np.float32)

    def store(self, texts: Sequence[str], embeddings: npt.NDArray[np.float32]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        final = self._directory / f"{self.key(texts)}.npy"
        tmp = final.with_name(f"{final.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, embeddings, allow_pickle=False)
        tmp.replace(final)  # atomic: a concurrent reader never sees half a file


class SiglipClassifier:
    model_name = MODEL_NAME

    def __init__(
        self,
        vision: Session,
        label_embeddings: npt.NDArray[np.float32],
        logit_scale: float = LOGIT_SCALE,
    ) -> None:
        self._vision = vision
        self._labels = normalize(label_embeddings)
        self._scale = logit_scale
        self.num_classes = len(label_embeddings)

    @classmethod
    def from_dir(
        cls, model_dir: Path, taxonomy: Taxonomy, cache_dir: Path, threads: int = 0
    ) -> "SiglipClassifier":
        texts = [PROMPT_TEMPLATE.format(label.text) for label in taxonomy.labels]
        # Two labels with one text get one embedding and split its probability; that is a
        # taxonomy bug, so it fails startup instead of degrading to 503.
        duplicates = sorted(text for text, count in Counter(texts).items() if count > 1)
        if duplicates:
            raise TaxonomyError(
                f"labels share a text (set a distinct name or prompt): {duplicates}"
            )
        missing = [
            f for f in (VISION_FILE, TEXT_FILE, TOKENIZER_FILE) if not (model_dir / f).is_file()
        ]
        if missing:
            raise ModelNotAvailableError(
                f"SigLIP files missing in {model_dir} ({', '.join(missing)}); "
                "run scripts/fetch_model.py"
            )
        vision = _session(model_dir / VISION_FILE, threads)
        cache = LabelEmbeddingCache(cache_dir, _fingerprint(model_dir))
        embeddings = cache.load(texts, width=_embedding_width(vision))
        if embeddings is None:
            logger.info("embedding %d labels with the SigLIP text tower", len(texts))
            encoder = TextEncoder(
                Tokenizer.from_file(str(model_dir / TOKENIZER_FILE)),
                lambda: _session(model_dir / TEXT_FILE, threads),
            )
            embeddings = encoder.embed(texts)
            encoder.close()
            cache.store(texts, embeddings)
        return cls(vision, embeddings)

    def embed_image(self, data: bytes) -> npt.NDArray[np.float32]:
        (pooled,) = self._vision.run(["pooler_output"], {"pixel_values": preprocess(data)})
        return normalize(np.asarray(pooled, dtype=np.float32)[0])

    def predict(self, data: bytes) -> npt.NDArray[np.float32]:
        cosine = self._labels @ self.embed_image(data)
        return softmax((self._scale * cosine).astype(np.float32))


def _session(path: Path, threads: int = 0) -> Session:
    session: Session = load_session(path, threads)
    return session


def _fingerprint(model_dir: Path) -> str:
    """Tokenizer content hash + text tower size: changes when either file is swapped."""
    tokenizer = hashlib.sha256((model_dir / TOKENIZER_FILE).read_bytes()).hexdigest()
    return f"{tokenizer}:{(model_dir / TEXT_FILE).stat().st_size}"


def _embedding_width(vision: Session) -> int | None:
    outputs = getattr(vision, "get_outputs", None)
    if outputs is None:
        return None
    width = next((o.shape[-1] for o in outputs() if o.name == "pooler_output"), None)
    return width if isinstance(width, int) else None


def _token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ModelNotAvailableError(f"tokenizer has no {token} token")
    return int(token_id)
