"""Read and validate untrusted image uploads."""

import io
from dataclasses import dataclass

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
