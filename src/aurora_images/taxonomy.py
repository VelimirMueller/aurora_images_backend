"""Topic tree over the classifier's labels, and roll-up of label probabilities into topics."""

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml


class TaxonomyError(Exception):
    pass


@dataclass(frozen=True)
class Topic:
    id: str
    parent: str | None
    depth: int
    path: tuple[str, ...]  # root topic ... this topic


@dataclass(frozen=True)
class Label:
    id: str
    name: str
    topic: str
    index: int  # position in the classifier's output vector


class Taxonomy:
    def __init__(self, topics: list[dict[str, Any]], labels: list[dict[str, Any]], version: int):
        self.version = version
        self.topics = _build_topics(topics)
        self.labels = _build_labels(labels, self.topics)
        # membership[t, l] = 1 when label l sits under topic t or any of its descendants.
        self._topic_ids = list(self.topics)
        row = {topic_id: i for i, topic_id in enumerate(self._topic_ids)}
        self._membership = np.zeros((len(self._topic_ids), len(self.labels)), dtype=np.float32)
        for label in self.labels:
            for topic_id in self.topics[label.topic].path:
                self._membership[row[topic_id], label.index] = 1.0

    @classmethod
    def load(cls, path: Path | None = None) -> "Taxonomy":
        """Load a taxonomy YAML file, or the packaged default when path is None.

        Every failure (unreadable file, bad YAML, missing fields) surfaces as TaxonomyError.
        """
        source = str(path) if path else "packaged taxonomy.yaml"
        try:
            if path is None:
                text = resources.files("aurora_images.data").joinpath("taxonomy.yaml").read_text()
            else:
                text = path.read_text(encoding="utf-8")
            document = yaml.safe_load(text)
        except (OSError, yaml.YAMLError) as exc:
            raise TaxonomyError(f"cannot read {source}: {exc}") from exc
        if not isinstance(document, dict) or not {"topics", "labels"} <= document.keys():
            raise TaxonomyError(f"{source} must be a mapping with 'topics' and 'labels'")
        try:
            return cls(document["topics"], document["labels"], int(document.get("version", 1)))
        except (KeyError, TypeError, ValueError) as exc:
            raise TaxonomyError(f"{source} has a malformed entry: {exc!r}") from exc

    def label_path(self, label: Label) -> list[str]:
        return [*self.topics[label.topic].path, label.name]

    def rollup(self, probabilities: npt.NDArray[np.float32]) -> dict[str, float]:
        """Topic score = sum of the probabilities of every label below it."""
        if probabilities.shape != (len(self.labels),):
            raise TaxonomyError(
                f"expected {len(self.labels)} probabilities, got shape {probabilities.shape}"
            )
        scores = self._membership @ probabilities
        return {
            topic_id: float(score) for topic_id, score in zip(self._topic_ids, scores, strict=True)
        }


def _build_topics(raw: list[dict[str, Any]]) -> dict[str, Topic]:
    parents: dict[str, str | None] = {}
    for item in raw:
        topic_id = item["id"]
        if topic_id in parents:
            raise TaxonomyError(f"duplicate topic id: {topic_id}")
        parents[topic_id] = item.get("parent")

    topics: dict[str, Topic] = {}
    for topic_id in parents:
        path: list[str] = []
        current: str | None = topic_id
        while current is not None:
            if current not in parents:
                raise TaxonomyError(f"topic {path[-1]!r} has unknown parent {current!r}")
            if current in path:
                raise TaxonomyError(f"cycle in topic tree at {current!r}")
            path.append(current)
            current = parents[current]
        topics[topic_id] = Topic(topic_id, parents[topic_id], len(path) - 1, tuple(reversed(path)))
    return topics


def _build_labels(raw: list[dict[str, Any]], topics: dict[str, Topic]) -> list[Label]:
    labels = [Label(item["id"], item["name"], item["topic"], item["index"]) for item in raw]
    # bool is an int subclass and 1.0 == 1, so both would slip through the range check below.
    if any(type(label.index) is not int for label in labels):
        raise TaxonomyError("label indices must be integers")
    if not labels:
        raise TaxonomyError("taxonomy has no labels")
    if sorted(label.index for label in labels) != list(range(len(labels))):
        raise TaxonomyError("label indices must be unique and cover 0..n-1")
    if len({label.id for label in labels}) != len(labels):
        raise TaxonomyError("duplicate label id")
    for label in labels:
        if label.topic not in topics:
            raise TaxonomyError(f"label {label.id} references unknown topic {label.topic!r}")
    return sorted(labels, key=lambda label: label.index)
