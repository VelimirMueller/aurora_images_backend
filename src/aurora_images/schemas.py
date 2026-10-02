from pydantic import BaseModel, Field


class ImageOut(BaseModel):
    id: str = Field(description="Server-generated image id")
    content_type: str
    size_bytes: int
    width: int
    height: int


class ImageInfo(BaseModel):
    sha256: str
    content_type: str
    size_bytes: int
    width: int
    height: int


class ModelInfo(BaseModel):
    name: str
    taxonomy_version: int


class LabelOut(BaseModel):
    id: str = Field(description="Stable label id (WordNet synset id for ImageNet labels)")
    label: str
    score: float = Field(ge=0, le=1, description="Probability of this exact label")
    topic: str = Field(description="Most specific topic the label belongs to")
    path: list[str] = Field(description="Root topic → … → topic → label")


class TopicOut(BaseModel):
    id: str
    parent: str | None
    depth: int = Field(description="0 for a root topic such as 'animal' or 'vehicle'")
    score: float = Field(ge=0, le=1, description="Sum of the probabilities of all labels below")


class ClassificationOut(BaseModel):
    request_id: str
    image: ImageInfo
    model: ModelInfo
    primary: LabelOut = Field(description="Most likely label")
    topic: TopicOut = Field(description="Most likely root topic (the meta topic)")
    labels: list[LabelOut] = Field(description="Top-k labels, best first")
    topics: list[TopicOut] = Field(description="Topics above the minimum score, best first")
    uncertain: bool = Field(description="True when the root topic score is below the threshold")
    timings_ms: dict[str, float]


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    model: str | None = Field(description="Loaded model name, or null when classification is off")


class ErrorOut(BaseModel):
    detail: str
