"""Image storage. The API depends on the protocol, so an object store can replace the disk."""

import os
import uuid
from pathlib import Path
from typing import Protocol


class ImageStorage(Protocol):
    def save(self, data: bytes, extension: str) -> str:
        """Persist data and return the server-generated image id."""
        ...


class LocalImageStorage:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def save(self, data: bytes, extension: str) -> str:
        image_id = uuid.uuid4().hex
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{image_id}.{extension}"
        # O_EXCL: never overwrite an existing file, even on a uuid collision.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return image_id
