"""Embed the configured taxonomy's labels now, so the server starts in under a second.

Runs at Docker build time. Uses the same settings (SYNTHWERK_*) as the server.
Usage: uv run python scripts/warm_label_cache.py
"""

from synthwerk_vision.config import get_settings
from synthwerk_vision.main import build_service


def main() -> None:
    settings = get_settings()
    service = build_service(settings)
    if service is None:
        raise SystemExit("model not available; run scripts/fetch_model.py")
    print(f"{service.model_name}: {len(service.taxonomy.labels)} labels ready")


if __name__ == "__main__":
    main()
