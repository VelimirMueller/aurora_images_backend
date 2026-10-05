"""Download the pretrained model and labels into models/, verified by SHA-256.

Weights are not committed to git. Usage: uv run python scripts/fetch_model.py
"""

import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ARTIFACTS = {
    "mobilenetv2-12.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
        "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5",
    ),
    "imagenet_classes.txt": (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "1f386e0d1cb6e28b9c2dac651c3dea6801e98ad1b41a14ce6bb1a093d72069f5",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: Path, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as fh:  # noqa: S310 - fixed https URLs
                shutil.copyfileobj(response, fh)
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == attempts:
                sys.exit(f"download failed after {attempts} attempts: {url}: {exc}")
            time.sleep(2**attempt)


def fetch(name: str, url: str, expected: str) -> None:
    target = MODELS_DIR / name
    if target.is_file() and sha256(target) == expected:
        print(f"ok        {name}")
        return
    tmp = target.with_suffix(target.suffix + ".part")
    download(url, tmp)
    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink()
        sys.exit(f"checksum mismatch for {name}: expected {expected}, got {actual}")
    tmp.replace(target)
    print(f"fetched   {name}")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, (url, digest) in ARTIFACTS.items():
        fetch(name, url, digest)


if __name__ == "__main__":
    main()
