"""Score a classification service against a manifest of images with accepted root topics."""

import hashlib
import math
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import UnidentifiedImageError

from aurorae_images.images import dhash, dhash_distance
from aurorae_images.service import Classification, ClassificationService

USER_AGENT = "aurorae-images-eval/0.1 (https://github.com/VelimirMueller/aurorae_images_backend)"
# Wikimedia renders thumbnails per data centre, so the same picture can arrive with other bytes.
# A 64-bit difference hash tells "same picture, re-encoded" from "different picture":
# re-encodes/resizes of the 58 eval images differ by at most 6 bits, distinct images by >= 15.
DHASH_MAX_DISTANCE = 8


@dataclass(frozen=True)
class Item:
    title: str
    url: str
    sha256: str
    topics: tuple[str, ...]  # accepted root topics; the first is the main one
    license: str
    author: str
    dhash: str | None = None  # hex; lets a re-encoded but identical picture through


@dataclass(frozen=True)
class Outcome:
    title: str
    expected: tuple[str, ...]
    predicted: str
    top3: tuple[str, ...]
    primary: str
    score: float
    latency_ms: float

    @property
    def hit(self) -> bool:
        return self.predicted in self.expected

    @property
    def hit_top3(self) -> bool:
        return any(topic in self.expected for topic in self.top3)


@dataclass
class Report:
    model: str
    images: int
    unavailable: list[str] = field(default_factory=list)
    reencoded: list[str] = field(default_factory=list)  # same picture, other bytes upstream
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def scored(self) -> int:
        return len(self.outcomes)

    @property
    def available_ratio(self) -> float:
        return self.scored / self.images if self.images else 0.0

    @property
    def topic_accuracy(self) -> float:
        return _ratio(sum(o.hit for o in self.outcomes), self.scored)

    def topic_accuracy_interval(self, z: float = 1.96) -> tuple[float, float]:
        """95 % Wilson score interval for the root-topic accuracy."""
        n = self.scored
        if n == 0:
            return 0.0, 0.0
        p = self.topic_accuracy
        denominator = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denominator
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        return round(centre - half, 3), round(centre + half, 3)

    @property
    def topic_accuracy_top3(self) -> float:
        return _ratio(sum(o.hit_top3 for o in self.outcomes), self.scored)

    def per_topic(self) -> dict[str, tuple[int, int]]:
        """Main expected topic -> (hits, total)."""
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for outcome in self.outcomes:
            counts[outcome.expected[0]][0] += outcome.hit
            counts[outcome.expected[0]][1] += 1
        return {topic: (hits, total) for topic, (hits, total) in sorted(counts.items())}

    def latency_ms(self) -> tuple[float, float]:
        """(mean, p95) per image, including preprocessing."""
        if not self.outcomes:
            return 0.0, 0.0
        values = np.array([o.latency_ms for o in self.outcomes])
        return round(float(values.mean()), 1), round(float(np.percentile(values, 95)), 1)

    def to_dict(self) -> dict[str, Any]:
        mean, p95 = self.latency_ms()
        return {
            "model": self.model,
            "images": self.images,
            "scored": self.scored,
            "unavailable": self.unavailable,
            "reencoded": self.reencoded,
            "topic_accuracy": round(self.topic_accuracy, 4),
            "topic_accuracy_95ci": list(self.topic_accuracy_interval()),
            "topic_accuracy_top3": round(self.topic_accuracy_top3, 4),
            "latency_ms": {"mean": mean, "p95": p95},
            "per_topic": {t: {"hits": h, "total": n} for t, (h, n) in self.per_topic().items()},
            "outcomes": [{**asdict(o), "hit": o.hit} for o in self.outcomes],
        }

    def markdown(self) -> str:
        mean, p95 = self.latency_ms()
        low, high = self.topic_accuracy_interval()
        lines = [
            f"## Eval: {self.model}",
            "",
            f"- Root-topic accuracy: **{self.topic_accuracy:.1%}** "
            f"({sum(o.hit for o in self.outcomes)}/{self.scored}, 95 % CI "
            f"{low:.1%}-{high:.1%}), in top 3: {self.topic_accuracy_top3:.1%}",
            f"- Images scored: {self.scored}/{self.images}"
            + (f" ({len(self.unavailable)} unavailable)" if self.unavailable else "")
            + (
                f", {len(self.reencoded)} re-encoded upstream (same picture by dHash)"
                if self.reencoded
                else ""
            ),
            f"- Latency per image: mean {mean} ms, p95 {p95} ms",
            "",
            "| Topic | Hits |",
            "|---|---|",
            *(f"| {t} | {h}/{n} |" for t, (h, n) in self.per_topic().items()),
        ]
        misses = [o for o in self.outcomes if not o.hit]
        if misses:
            lines += ["", "| Miss | Expected | Predicted (primary label) |", "|---|---|---|"]
            lines += [
                f"| {o.title.removeprefix('File:')[:60]} | {' / '.join(o.expected)} "
                f"| {o.predicted} ({o.primary} {o.score:.2f}) |"
                for o in misses
            ]
        return "\n".join(lines) + "\n"


def load_manifest(path: Path) -> list[Item]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Item(
            title=entry["title"],
            url=entry["url"],
            sha256=entry["sha256"],
            topics=tuple(entry["topics"]),
            license=entry["license"],
            author=entry["author"],
            dhash=entry.get("dhash"),
        )
        for entry in document["images"]
    ]


class Unavailable(Exception):
    """The image cannot be scored; the message says why (shown in the report)."""


_RETRY_STATUS = {429, 500, 502, 503, 504}


def same_picture(data: bytes, item: Item) -> bool:
    if item.dhash is None:
        return False
    try:
        actual = dhash(data)
    except (UnidentifiedImageError, OSError):
        return False  # not even an image (truncated download, error page)
    return dhash_distance(actual, item.dhash) <= DHASH_MAX_DISTANCE


def fetch(item: Item, cache_dir: Path, *, attempts: int = 4, pause: float = 0.3) -> bytes:
    """Return the verified image bytes, or raise Unavailable with the reason.

    Only verified bytes are cached. A cached file with a wrong hash (truncated write, old
    state) is removed and downloaded again, so it cannot stay "unavailable" forever. Rate
    limits and server errors are retried with backoff (Commons throttles bursts from CI IPs).
    """
    path = cache_dir / f"{item.sha256[:16]}.jpg"
    if path.is_file():
        data = path.read_bytes()
        if _verified(data, item) or same_picture(data, item):
            return data
        path.unlink()
    downloaded = _download(item.url, attempts, pause)
    if not (_verified(downloaded, item) or same_picture(downloaded, item)):
        raise Unavailable("upstream file changed (SHA-256 and dHash mismatch)")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(downloaded)
    return downloaded


def _download(url: str, attempts: int, pause: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    for attempt in range(1, attempts + 1):
        time.sleep(pause)  # pace requests; never a burst
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                data: bytes = response.read()
                return data
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRY_STATUS or attempt == attempts:
                raise Unavailable(f"HTTP {exc.code}") from exc
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            time.sleep(float(retry_after) if retry_after.isdigit() else 2.0**attempt)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == attempts:
                raise Unavailable(f"network error: {exc}") from exc
            time.sleep(2.0**attempt)
    raise Unavailable("no attempts made")


def _verified(data: bytes, item: Item) -> bool:
    return hashlib.sha256(data).hexdigest() == item.sha256


def outcome(item: Item, result: Classification, latency_ms: float) -> Outcome:
    roots = [t.id for t in sorted(result.topics, key=lambda t: -t.score) if t.parent is None]
    return Outcome(
        title=item.title,
        expected=item.topics,
        predicted=result.topic.id,
        top3=tuple(roots[:3]) or (result.topic.id,),
        primary=result.labels[0].label.name,
        score=result.labels[0].score,
        latency_ms=round(latency_ms, 1),
    )


def evaluate(service: ClassificationService, items: list[Item], cache_dir: Path) -> Report:
    report = Report(model=service.model_name, images=len(items))
    for item in items:
        try:
            data = fetch(item, cache_dir)
        except Unavailable as exc:
            report.unavailable.append(f"{item.title} ({exc})")
            continue
        if not _verified(data, item):
            report.reencoded.append(item.title)
        started = time.perf_counter()
        result = service.classify(data)
        report.outcomes.append(outcome(item, result, (time.perf_counter() - started) * 1000))
    return report


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else 0.0
