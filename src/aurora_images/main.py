import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aurora_images import __version__
from aurora_images.api.routes import router
from aurora_images.classifier import Classifier, ModelNotAvailableError, OnnxClassifier
from aurora_images.config import Settings, get_settings
from aurora_images.storage import ImageStorage, LocalImageStorage

logger = logging.getLogger("aurora_images")


def _load_classifier(settings: Settings) -> Classifier | None:
    try:
        return OnnxClassifier(settings.model_path, settings.labels_path)
    except ModelNotAvailableError as exc:
        logger.warning("classification disabled: %s", exc)
        return None


def create_app(
    settings: Settings | None = None,
    *,
    storage: ImageStorage | None = None,
    classifier: Classifier | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.storage = storage or LocalImageStorage(settings.upload_dir)
        app.state.classifier = classifier or _load_classifier(settings)
        yield

    app = FastAPI(title="Aurora Images API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    return app
