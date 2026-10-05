from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables prefixed with AURORA_."""

    model_config = SettingsConfigDict(env_prefix="AURORA_", env_file=".env", extra="ignore")

    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:8080"],
        description="Browser origins allowed to call the API. Never use '*' in production.",
    )
    upload_dir: Path = Path("uploads")
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=40_000_000, gt=0)
    model_path: Path = Path("models/mobilenetv2-12.onnx")
    labels_path: Path = Path("models/imagenet_classes.txt")
    top_k: int = Field(default=5, ge=1, le=20)


@lru_cache
def get_settings() -> Settings:
    return Settings()
