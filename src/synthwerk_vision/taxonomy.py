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
    prompt: str | None = None  # text an open-vocabulary model embeds; defaults to name

    @property
    def text(self) -> str:
        return self.prompt or self.name


class Taxonomy:
    def __init__(self, topics: list[dict[str, Any]], labels: list[dict[str, Any]], version: int):
        self.version = version
        self.topics = _build_topics(topics)
        self.labels = _build_labels(labels, self.topics)
        self._by_id = {label.id: label for label in self.labels}
        # membership[t, l] = 1 when label l sits under topic t or any of its descendants.
        self._topic_ids = list(self.topics)
        row = {topic_id: i for i, topic_id in enumerate(self._topic_ids)}
        self._membership = np.zeros((len(self._topic_ids), len(self.labels)), dtype=np.float32)
        for label in self.labels:
            for topic_id in self.topics[label.topic].path:
                self._membership[row[topic_id], label.index] = 1.0

    @classmethod
    def load(cls, path: Path | None = None, *, packaged: str = "taxonomy.yaml") -> Taxonomy:
        """Load a taxonomy YAML file, or the packaged file named `packaged` when path is None.

        A file may `extends:` another one (`package:<name>` or a path relative to itself) and
        add `prompts`, `topics` and `labels` on top. Every failure (unreadable file, bad YAML,
        missing fields, broken extends chain) surfaces as TaxonomyError.
        """
        source = _Source(path, packaged)
        document = _resolve(source, seen=set())
        try:
            return cls(document["topics"], document["labels"], int(document.get("version", 1)))
        except (KeyError, TypeError, ValueError) as exc:
            raise TaxonomyError(f"{source} has a malformed entry: {exc!r}") from exc

    def label(self, label_id: str) -> Label | None:
        return self._by_id.get(label_id)

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
    labels = [
        Label(item["id"], item["name"], item["topic"], item["index"], item.get("prompt"))
        for item in raw
    ]
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


_PACKAGE_PREFIX = "package:"
_MAX_EXTENDS_DEPTH = 4


@dataclass(frozen=True)
class _Source:
    path: Path | None  # None = a packaged file
    packaged: str

    def __str__(self) -> str:
        return str(self.path) if self.path else f"packaged {self.packaged}"

    def read(self) -> dict[str, Any]:
        try:
            if self.path is None:
                text = resources.files("synthwerk_vision.data").joinpath(self.packaged).read_text()
            else:
                text = self.path.read_text(encoding="utf-8")
            document = yaml.safe_load(text)
        except (OSError, yaml.YAMLError) as exc:
            raise TaxonomyError(f"cannot read {self}: {exc}") from exc
        if not isinstance(document, dict):
            raise TaxonomyError(f"{self} must be a mapping with 'topics' and 'labels'")
        return document

    def base(self, reference: object) -> _Source:
        if not isinstance(reference, str) or not reference:
            raise TaxonomyError(f"{self}: 'extends' must be a file name")
        if reference.startswith(_PACKAGE_PREFIX):
            return _Source(None, reference.removeprefix(_PACKAGE_PREFIX))
        if self.path is None:
            return _Source(None, reference)  # packaged files extend packaged siblings
        return _Source((self.path.parent / reference).resolve(), "")


def _resolve(source: _Source, seen: set[str]) -> dict[str, Any]:
    """Read a document and, if it extends another, merge it on top of its resolved base."""
    if str(source) in seen:
        raise TaxonomyError(f"'extends' cycle through {source}")
    if len(seen) >= _MAX_EXTENDS_DEPTH:
        raise TaxonomyError(f"'extends' chain deeper than {_MAX_EXTENDS_DEPTH} files at {source}")
    seen.add(str(source))
    document = source.read()
    if "extends" not in document:
        if not {"topics", "labels"} <= document.keys():
            raise TaxonomyError(f"{source} must be a mapping with 'topics' and 'labels'")
        return document
    base = _resolve(source.base(document["extends"]), seen)
    try:
        return _merge(base, document)
    except (KeyError, TypeError) as exc:
        raise TaxonomyError(f"{source} has a malformed entry: {exc!r}") from exc


def _merge(base: dict[str, Any], extension: dict[str, Any]) -> dict[str, Any]:
    prompts = extension.get("prompts") or {}
    if not isinstance(prompts, dict):
        raise TaxonomyError("'prompts' must map label ids to text")
    base_labels: list[dict[str, Any]] = base["labels"]
    unknown = sorted(set(prompts) - {label["id"] for label in base_labels})
    if unknown:
        raise TaxonomyError(f"prompts for unknown label ids: {unknown}")
    labels = [
        {**label, "prompt": prompts[label["id"]]} if label["id"] in prompts else label
        for label in base_labels
    ]
    for offset, label in enumerate(extension.get("labels") or []):
        if "index" in label:
            raise TaxonomyError(
                f"label {label.get('id')!r}: extending labels get their index automatically"
            )
        labels.append({**label, "index": len(base_labels) + offset})
    return {
        "version": extension.get("version", base.get("version", 1)),
        "topics": [*base["topics"], *(extension.get("topics") or [])],
        "labels": labels,
    }
