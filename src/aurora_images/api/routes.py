from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from aurora_images.classifier import Classifier
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
    ImageOut,
    PredictionOut,
)
from aurora_images.storage import ImageStorage

router = APIRouter()

_IMAGE_ERRORS: dict[int | str, dict[str, object]] = {
    413: {"model": ErrorOut, "description": "Upload exceeds the size limit"},
    415: {"model": ErrorOut, "description": "Not a JPEG, PNG or WebP image"},
}


def get_storage(request: Request) -> ImageStorage:
    storage: ImageStorage = request.app.state.storage
    return storage


def get_classifier(request: Request) -> Classifier:
    classifier: Classifier | None = request.app.state.classifier
    if classifier is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "classification model not loaded")
    return classifier


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
    return HealthOut(status="ok", model_loaded=request.app.state.classifier is not None)


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
    image: Annotated[ValidatedImage, Depends(validated_image)],
    classifier: Annotated[Classifier, Depends(get_classifier)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ClassificationOut:
    """Classify an image without storing it. Returns the top-k labels with probabilities."""
    predictions = await run_in_threadpool(classifier.classify, image.data, settings.top_k)
    return ClassificationOut(
        model=classifier.model_name,
        predictions=[PredictionOut(label=p.label, score=p.score) for p in predictions],
    )
