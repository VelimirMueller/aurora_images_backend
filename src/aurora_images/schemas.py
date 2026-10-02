from pydantic import BaseModel, Field


class ImageOut(BaseModel):
    id: str = Field(description="Server-generated image id")
    content_type: str
    size_bytes: int
    width: int
    height: int


class PredictionOut(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


class ClassificationOut(BaseModel):
    model: str
    predictions: list[PredictionOut]


class HealthOut(BaseModel):
    status: str
    model_loaded: bool


class ErrorOut(BaseModel):
    detail: str
