"""Read and validate untrusted image uploads."""

import io
from dataclasses import dataclass

import numpy as np
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

# Decoded Pillow format -> file extension. The extension never comes from the client.
ALLOWED_FORMATS: dict[str, str] = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}

_CHUNK_SIZE = 64 * 1024


class UploadTooLargeError(Exception):
    pass


class InvalidImageError(Exception):
    pass


@dataclass(frozen=True)
class ValidatedImage:
    data: bytes
    format: str
    extension: str
    width: int
    height: int


async def read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read the upload in chunks and stop as soon as it exceeds max_bytes."""
    buffer = bytearray()
    while chunk := await upload.read(_CHUNK_SIZE):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise UploadTooLargeError(f"upload exceeds {max_bytes} bytes")
    return bytes(buffer)


def validate_image(data: bytes, max_pixels: int) -> ValidatedImage:
    """Check the magic bytes and dimensions by decoding the header, not by trusting Content-Type."""
    if not data:
        raise InvalidImageError("empty upload")
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format or ""
            width, height = img.size
            if fmt not in ALLOWED_FORMATS:
                raise InvalidImageError(f"unsupported image format: {fmt or 'unknown'}")
            if width * height > max_pixels:
                raise InvalidImageError("image dimensions too large")
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("file is not a valid image") from exc
    return ValidatedImage(data, fmt, ALLOWED_FORMATS[fmt], width, height)


def dhash(data: bytes) -> str:
    """64-bit difference hash of the picture (grayscale 9x8, left > right), as hex."""
    with Image.open(io.BytesIO(data)) as image:
        small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, :-1] > pixels[:, 1:]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def dhash_distance(a: str, b: str) -> int:
    """Number of differing bits between two hex dHashes."""
    return (int(a, 16) ^ int(b, 16)).bit_count()
