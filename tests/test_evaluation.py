import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from aurora_images.evaluation import Item, Outcome, Report, evaluate, fetch, load_manifest
from aurora_images.service import ClassificationService
from aurora_images.taxonomy import Taxonomy
from tests.conftest import SMALL_TAXONOMY, FakeClassifier, make_image


def small_service(probabilities: list[float] | None = None) -> ClassificationService:
    document = copy.deepcopy(SMALL_TAXONOMY)
    taxonomy = Taxonomy(document["topics"], document["labels"], document["version"])
    return ClassificationService(
        FakeClassifier(probabilities), taxonomy, top_k=3, topic_min_score=0.0, uncertain_below=0.5
    )


def cached_item(tmp_path: Path, topics: tuple[str, ...], data: bytes | None = None) -> Item:
    data = data if data is not None else make_image()
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / f"{digest[:16]}.jpg").write_bytes(data)
    return Item("File:x.jpg", "https://invalid.example/x.jpg", digest, topics, "CC0", "someone")


def outcome(expected: tuple[str, ...], predicted: str, top3: tuple[str, ...] = ()) -> Outcome:
    return Outcome("File:t.jpg", expected, predicted, top3 or (predicted,), "label", 0.5, 10.0)


def test_report_metrics_count_any_accepted_topic() -> None:
    report = Report(model="m", images=5)
    report.outcomes = [
        outcome(("animal",), "animal"),
        outcome(("person", "animal"), "animal"),  # second accepted topic counts as a hit
        outcome(("plant",), "object", ("object", "plant")),  # miss, but in top 3
        outcome(("vehicle",), "object", ("object", "landscape")),
    ]
    report.unavailable = ["File:gone.jpg"]

    assert report.topic_accuracy == 0.5
    assert report.topic_accuracy_top3 == 0.75
    assert report.available_ratio == 0.8
    assert report.per_topic() == {
        "animal": (1, 1),
        "person": (1, 1),
        "plant": (0, 1),
        "vehicle": (0, 1),
    }
    assert report.latency_ms() == (10.0, 10.0)
    markdown = report.markdown()
    assert "**50.0%** (2/4)" in markdown
    assert "1 unavailable" in markdown
    assert "| x.jpg" not in markdown and "| t.jpg | vehicle | object" in markdown


def test_empty_report_does_not_divide_by_zero() -> None:
    report = Report(model="m", images=0)

    assert (report.topic_accuracy, report.available_ratio, report.latency_ms()) == (
        0.0,
        0.0,
        (0.0, 0.0),
    )


def test_evaluate_scores_cached_images_without_network(tmp_path: Path) -> None:
    hit = cached_item(tmp_path, ("animal",))
    miss = cached_item(tmp_path, ("vehicle",), make_image("JPEG"))

    report = evaluate(small_service(), [hit, miss], tmp_path)

    assert [o.hit for o in report.outcomes] == [True, False]
    assert report.outcomes[0].primary == "beagle"
    assert report.outcomes[0].top3 == ("animal", "vehicle", "landscape"), "roots by score"
    assert report.to_dict()["topic_accuracy"] == 0.5


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.data


def test_changed_upstream_file_is_unavailable_and_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = cached_item(tmp_path, ("animal",))
    cached = tmp_path / f"{item.sha256[:16]}.jpg"
    cached.write_bytes(b"truncated earlier download")
    monkeypatch.setattr(
        "aurora_images.evaluation.urllib.request.urlopen",
        lambda *a, **k: FakeResponse(b"re-rendered thumbnail"),
    )

    report = evaluate(small_service(), [item], tmp_path)

    assert report.unavailable == ["File:x.jpg"]
    assert report.scored == 0
    assert not cached.exists(), "a mismatched file must not stay in the cache"


def test_bad_cache_entry_is_replaced_by_a_verified_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = make_image()
    item = cached_item(tmp_path, ("animal",), good)
    (tmp_path / f"{item.sha256[:16]}.jpg").write_bytes(b"partial")
    monkeypatch.setattr(
        "aurora_images.evaluation.urllib.request.urlopen", lambda *a, **k: FakeResponse(good)
    )

    assert fetch(item, tmp_path) == good
    assert (tmp_path / f"{item.sha256[:16]}.jpg").read_bytes() == good


def test_frozen_manifest_is_complete_and_licensed() -> None:
    manifest = Path(__file__).resolve().parent.parent / "eval" / "manifest.yaml"
    items = load_manifest(manifest)
    roots = {
        t["id"]
        for t in yaml.safe_load(
            (
                Path(__file__).resolve().parent.parent / "src/aurora_images/data/taxonomy_open.yaml"
            ).read_text()
        )["topics"]
        if t["parent"] is None
    } | {t.id for t in Taxonomy.load().topics.values() if t.parent is None}

    assert len(items) == 58
    assert len({item.sha256 for item in items}) == len(items)
    assert all(item.license and item.author for item in items)
    assert all(set(item.topics) <= roots for item in items), "expected topics must be root topics"


@pytest.mark.parametrize("topics", [("animal",), ("person", "animal")])
def test_outcome_hit_logic(topics: tuple[str, ...]) -> None:
    assert outcome(topics, "animal").hit
