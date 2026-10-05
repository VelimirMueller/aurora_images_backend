"""Embed the configured taxonomy's labels now, so the server starts in under a second.

Runs at Docker build time. Uses the same settings (AURORA_*) as the server.
Usage: uv run python scripts/warm_label_cache.py
"""

from aurora_images.config import get_settings
from aurora_images.main import PACKAGED_TAXONOMY
from aurora_images.siglip import SiglipClassifier
from aurora_images.taxonomy import Taxonomy


def main() -> None:
    settings = get_settings()
    if settings.backend != "siglip":
        print(f"backend {settings.backend} has no label cache; nothing to do")
        return
    taxonomy = Taxonomy.load(settings.taxonomy_path, packaged=PACKAGED_TAXONOMY["siglip"])
    SiglipClassifier.from_dir(settings.siglip_dir, taxonomy, settings.label_cache_dir)
    print(f"label cache ready for {len(taxonomy.labels)} labels in {settings.label_cache_dir}")


if __name__ == "__main__":
    main()
