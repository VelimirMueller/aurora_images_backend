"""Download the pretrained model and labels into models/, verified by SHA-256.

Weights are not committed to git. URLs are pinned to a revision and every file is checked.

Usage: uv run python scripts/fetch_model.py [siglip] [mobilenet]   (default: both, ~1.5 GB)
"""

import argparse
import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SIGLIP = "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/ba1f3b0843f24bc5417d38e19c37b287d719b2f4"

# backend -> {path under models/: (url, sha256)}
ARTIFACTS = {
    "siglip": {
        "siglip2-base-patch16-224/vision_model.onnx": (
            f"{SIGLIP}/onnx/vision_model.onnx",
            "c0573e3f4140c3a7c4e9cc5912bd6b26a033b46a6a8e8af26cbea262b163bcad",
        ),
        "siglip2-base-patch16-224/text_model.onnx": (
            f"{SIGLIP}/onnx/text_model.onnx",
            "baf12d941beabafafb14f7b4adb38dc15be18681b964a84410ec53d9d65e6293",
        ),
        "siglip2-base-patch16-224/tokenizer.json": (
            f"{SIGLIP}/tokenizer.json",
            "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
        ),
    },
    "mobilenet": {
        "mobilenetv2-12.onnx": (
            "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
            "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5",
        ),
        # Display names and synset ids; only scripts/build_taxonomy.py reads these.
        "imagenet_classes.txt": (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            "1f386e0d1cb6e28b9c2dac651c3dea6801e98ad1b41a14ce6bb1a093d72069f5",
        ),
        "imagenet_class_index.json": (
            "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
            "a1e7a966a1f601d39e4b43e119b3e7dd4a2ad3ea08cf69847cbaf021013767bc",
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


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
    target.parent.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backends", nargs="*", choices=list(ARTIFACTS), metavar="backend")
    for backend in parser.parse_args().backends or ARTIFACTS:
        for name, (url, digest) in ARTIFACTS[backend].items():
            fetch(name, url, digest)


if __name__ == "__main__":
    main()
