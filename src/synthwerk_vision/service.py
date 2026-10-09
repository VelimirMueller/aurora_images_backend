"""Turn a classifier's probability vector into labels, topic paths and topic scores."""

import time
from dataclasses import dataclass

import numpy as np

from synthwerk_vision.classifier import Classifier
from synthwerk_vision.taxonomy import Label, Taxonomy, TaxonomyError


@dataclass(frozen=True)
class ScoredLabel:
    label: Label
    score: float
    path: list[str]


@dataclass(frozen=True)
class ScoredTopic:
    id: str
    parent: str | None
    depth: int
    score: float


@dataclass(frozen=True)
class Classification:
    labels: list[ScoredLabel]
    topics: list[ScoredTopic]
    topic: ScoredTopic  # best root topic ("meta topic")
    uncertain: bool
    timings_ms: dict[str, float]


class ClassificationService:
    def __init__(
        self,
        classifier: Classifier,
        taxonomy: Taxonomy,
        *,
        top_k: int,
        topic_min_score: float,
        uncertain_below: float,
    ) -> None:
        if classifier.num_classes != len(taxonomy.labels):
            raise TaxonomyError(
                f"taxonomy has {len(taxonomy.labels)} labels, "
                f"model {classifier.model_name} outputs {classifier.num_classes}"
            )
        self.classifier = classifier
        self.taxonomy = taxonomy
        self._top_k = top_k
        self._topic_min_score = topic_min_score
        self._uncertain_below = uncertain_below

    @property
    def model_name(self) -> str:
        return self.classifier.model_name

    @property
    def supports_runtime_labels(self) -> bool:
        return callable(getattr(self.classifier, "for_taxonomy", None))

    def with_taxonomy(self, taxonomy: Taxonomy) -> ClassificationService:
        """A new service for a changed label set; self keeps serving until it is swapped out."""
        if not self.supports_runtime_labels:
            raise TaxonomyError(f"{self.model_name} has a fixed label set")
        classifier: Classifier = self.classifier.for_taxonomy(taxonomy)  # type: ignore[attr-defined]
        return ClassificationService(
            classifier,
            taxonomy,
            top_k=self._top_k,
            topic_min_score=self._topic_min_score,
            uncertain_below=self._uncertain_below,
        )

    def classify(self, data: bytes) -> Classification:
        started = time.perf_counter()
        probabilities = self.classifier.predict(data)
        inferred = time.perf_counter()

        top = np.argsort(probabilities)[::-1][: self._top_k]
        labels = [
            ScoredLabel(
                self.taxonomy.labels[i],
                _round(probabilities[i]),
                self.taxonomy.label_path(self.taxonomy.labels[i]),
            )
            for i in top
        ]
        rolled = self.taxonomy.rollup(probabilities)
        all_topics = [
            ScoredTopic(t.id, t.parent, t.depth, _round(rolled[t.id]))
            for t in self.taxonomy.topics.values()
        ]
        best_root = max((t for t in all_topics if t.parent is None), key=lambda t: rolled[t.id])
        topics = sorted(
            (t for t in all_topics if rolled[t.id] >= self._topic_min_score),
            key=lambda t: (-t.score, t.depth),
        )
        done = time.perf_counter()

        return Classification(
            labels=labels,
            topics=topics,
            topic=best_root,
            uncertain=rolled[best_root.id] < self._uncertain_below,  # raw, not the rounded score
            timings_ms={
                "inference": _ms(started, inferred),
                "rollup": _ms(inferred, done),
            },
        )


def _round(value: float | np.floating) -> float:
    return round(float(value), 4)


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 2)
