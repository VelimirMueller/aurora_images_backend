import hashlib
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from aurora_images.config import Settings, get_settings
from aurora_images.images import (
    InvalidImageError,
    UploadTooLargeError,
    ValidatedImage,
    read_limited,
    validate_image,
)
from aurora_images.schemas import (
    ClassificationOut,
    ErrorOut,
    HealthOut,
    ImageInfo,
    ImageOut,
    LabelOut,
    ModelInfo,
    TopicOut,
)
from aurora_images.service import ClassificationService, ScoredLabel, ScoredTopic
from aurora_images.storage import ImageStorage

router = APIRouter()

_IMAGE_ERRORS: dict[int | str, dict[str, object]] = {
    413: {"model": ErrorOut, "description": "Upload exceeds the size limit"},
    415: {"model": ErrorOut, "description": "Not a JPEG, PNG or WebP image"},
}


def get_storage(request: Request) -> ImageStorage:
    storage: ImageStorage = request.app.state.storage
    return storage


def get_service(request: Request) -> ClassificationService:
    service: ClassificationService | None = request.app.state.service
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "classification model not loaded")
    return service


async def validated_image(
    image: Annotated[UploadFile, File(description="JPEG, PNG or WebP image")],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ValidatedImage:
    try:
        data = await read_limited(image, settings.max_upload_bytes)
        return validate_image(data, settings.max_image_pixels)
    except UploadTooLargeError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc


@router.get("/health", tags=["ops"])
def health(request: Request) -> HealthOut:
    return HealthOut(status="ok", model_loaded=request.app.state.service is not None)


@router.post(
    "/v1/images",
    status_code=status.HTTP_201_CREATED,
    tags=["images"],
    responses=_IMAGE_ERRORS,
)
async def upload_image(
    image: Annotated[ValidatedImage, Depends(validated_image)],
    storage: Annotated[ImageStorage, Depends(get_storage)],
) -> ImageOut:
    """Store an image under a server-generated id."""
    image_id = await run_in_threadpool(storage.save, image.data, image.extension)
    return ImageOut(
        id=image_id,
        content_type=f"image/{image.format.lower()}",
        size_bytes=len(image.data),
        width=image.width,
        height=image.height,
    )


@router.post(
    "/v1/classifications",
    tags=["classification"],
    responses={**_IMAGE_ERRORS, 503: {"model": ErrorOut, "description": "Model not loaded"}},
)
async def classify_image(
    request: Request,
    image: Annotated[ValidatedImage, Depends(validated_image)],
    service: Annotated[ClassificationService, Depends(get_service)],
) -> ClassificationOut:
    """Classify an image without storing it.

    Returns the top-k labels with their topic path (e.g. animal → mammal → dog → beagle) and
    the topic scores, where a topic's score is the summed probability of every label below it.
    """
    result = await run_in_threadpool(service.classify, image.data)
    total_ms = round((time.perf_counter() - request.state.started) * 1000, 2)
    return ClassificationOut(
        request_id=request.state.request_id,
        image=ImageInfo(
            sha256=hashlib.sha256(image.data).hexdigest(),
            content_type=f"image/{image.format.lower()}",
            size_bytes=len(image.data),
            width=image.width,
            height=image.height,
        ),
        model=ModelInfo(name=service.model_name, taxonomy_version=service.taxonomy.version),
        primary=_label_out(result.labels[0]),
        topic=_topic_out(result.topic),
        labels=[_label_out(label) for label in result.labels],
        topics=[_topic_out(topic) for topic in result.topics],
        uncertain=result.uncertain,
        timings_ms={**result.timings_ms, "total": total_ms},
    )


def _label_out(scored: ScoredLabel) -> LabelOut:
    return LabelOut(
        id=scored.label.id,
        label=scored.label.name,
        score=scored.score,
        topic=scored.label.topic,
        path=scored.path,
    )


def _topic_out(scored: ScoredTopic) -> TopicOut:
    return TopicOut(id=scored.id, parent=scored.parent, depth=scored.depth, score=scored.score)
