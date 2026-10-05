import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from aurora_images import __version__
from aurora_images.api.routes import router
from aurora_images.classifier import Classifier, ModelNotAvailableError, OnnxClassifier
from aurora_images.config import Settings, get_settings
from aurora_images.service import ClassificationService
from aurora_images.siglip import SiglipClassifier
from aurora_images.storage import ImageStorage, LocalImageStorage
from aurora_images.taxonomy import Taxonomy

logger = logging.getLogger("aurora_images")

# Accept a caller's X-Request-ID only when it is short and log-safe; otherwise mint one.
_REQUEST_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")


PACKAGED_TAXONOMY = {"siglip": "taxonomy_open.yaml", "mobilenet": "taxonomy.yaml"}


def _load_classifier(settings: Settings, taxonomy: Taxonomy) -> Classifier | None:
    try:
        if settings.backend == "siglip":
            return SiglipClassifier.from_dir(
                settings.siglip_dir, taxonomy, settings.label_cache_dir, settings.ort_threads
            )
        return OnnxClassifier(settings.model_path, settings.ort_threads)
    except ModelNotAvailableError as exc:
        logger.warning("classification disabled: %s", exc)
        return None


def _build_service(
    settings: Settings, classifier: Classifier | None
) -> ClassificationService | None:
    # A missing model is an ops state (503 + /health); a broken taxonomy is a config bug and
    # must stop startup, so TaxonomyError is not caught here.
    taxonomy = Taxonomy.load(settings.taxonomy_path, packaged=PACKAGED_TAXONOMY[settings.backend])
    classifier = classifier or _load_classifier(settings, taxonomy)
    if classifier is None:
        return None
    return ClassificationService(
        classifier,
        taxonomy,
        top_k=settings.top_k,
        topic_min_score=settings.topic_min_score,
        uncertain_below=settings.uncertain_below,
    )


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
        app.state.service = _build_service(settings, classifier)
        yield

    app = FastAPI(title="Aurora Images API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - request.state.started) * 1000,
        )
        return response

    app.include_router(router)
    return app
