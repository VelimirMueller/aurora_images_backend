from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    backend: Literal["siglip", "mobilenet"] = Field(
        default="siglip",
        description="siglip: open vocabulary (text labels). mobilenet: the 1000 ImageNet classes.",
    )
    siglip_dir: Path = Path("models/siglip2-base-patch16-224")
    label_cache_dir: Path = Path("models/label_cache")
    model_path: Path = Path("models/mobilenetv2-12.onnx")
    taxonomy_path: Path | None = Field(
        default=None,
        description="Topic tree YAML; None uses the packaged taxonomy for the backend.",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    topic_min_score: float = Field(default=0.05, ge=0, le=1)
    uncertain_below: float = Field(
        default=0.5, ge=0, le=1, description="Flag a result when the best root topic scores lower."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
