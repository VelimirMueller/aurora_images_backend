"""Generate src/aurora_images/data/taxonomy.yaml from WordNet.

Every ImageNet-1k class is a WordNet synset. Each class goes under the topic whose WordNet
anchor is the *closest* hypernym (breadth-first over all hypernym paths). The output is
committed and meant to be edited by hand afterwards; rerun only to start over.

Usage: uv run python scripts/build_taxonomy.py
"""

import json
from collections import Counter, deque
from pathlib import Path

import nltk
import yaml
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

ROOT = Path(__file__).resolve().parent.parent
CLASS_INDEX = ROOT / "models" / "imagenet_class_index.json"  # index -> [wnid, keras name]
DISPLAY_NAMES = ROOT / "models" / "imagenet_classes.txt"  # index -> human-readable name
OUTPUT = ROOT / "src" / "aurora_images" / "data" / "taxonomy.yaml"

# (topic id, parent topic id, WordNet anchors). Parents must be listed before children.
TOPICS: list[tuple[str, str | None, list[str]]] = [
    ("animal", None, ["animal.n.01"]),
    ("mammal", "animal", ["mammal.n.01"]),
    ("dog", "mammal", ["dog.n.01"]),
    ("wild canine", "mammal", ["canine.n.02"]),
    ("cat", "mammal", ["cat.n.01"]),
    ("big cat", "mammal", ["big_cat.n.01"]),
    ("primate", "mammal", ["primate.n.02"]),
    ("hoofed mammal", "mammal", ["ungulate.n.01"]),
    ("bear", "mammal", ["bear.n.01"]),
    ("rodent", "mammal", ["rodent.n.01"]),
    ("marine mammal", "mammal", ["aquatic_mammal.n.01"]),
    ("bird", "animal", ["bird.n.01"]),
    ("fish", "animal", ["fish.n.01"]),
    ("reptile", "animal", ["reptile.n.01"]),
    ("amphibian", "animal", ["amphibian.n.03"]),
    ("insect", "animal", ["insect.n.01"]),
    ("arachnid", "animal", ["arachnid.n.01"]),
    ("invertebrate", "animal", ["invertebrate.n.01"]),
    ("person", None, ["person.n.01"]),
    ("plant", None, ["plant.n.02", "plant_part.n.01"]),
    ("flower", "plant", ["flower.n.01"]),
    ("fungus", None, ["fungus.n.01"]),
    ("food and drink", None, ["food.n.01", "food.n.02"]),
    ("fruit", "food and drink", ["edible_fruit.n.01"]),
    ("vegetable", "food and drink", ["vegetable.n.01"]),
    ("dish", "food and drink", ["dish.n.02"]),
    ("drink", "food and drink", ["beverage.n.01"]),
    ("vehicle", None, ["vehicle.n.01", "conveyance.n.03"]),
    ("car", "vehicle", ["car.n.01"]),
    ("truck", "vehicle", ["truck.n.01"]),
    ("motor vehicle", "vehicle", ["motor_vehicle.n.01"]),
    ("bus", "vehicle", ["bus.n.01"]),
    ("bicycle", "vehicle", ["bicycle.n.01"]),
    ("rail vehicle", "vehicle", ["locomotive.n.01", "streetcar.n.01", "passenger_car.n.01"]),
    ("watercraft", "vehicle", ["vessel.n.02", "boat.n.01"]),
    ("aircraft", "vehicle", ["aircraft.n.01"]),
    ("building and structure", None, ["structure.n.01"]),
    (
        "building",
        "building and structure",
        [
            "building.n.01",
            "housing.n.01",
            "shelter.n.01",
            "shop.n.01",
            "mercantile_establishment.n.01",
            "fortification.n.01",
            "penal_institution.n.01",
            "factory.n.01",
        ],
    ),
    ("roof", "building and structure", ["roof.n.01"]),
    ("bridge", "building and structure", ["bridge.n.01"]),
    ("landscape", None, ["geological_formation.n.01", "body_of_water.n.01"]),
    (
        "clothing",
        None,
        [
            "clothing.n.01",
            "footwear.n.02",
            "headdress.n.01",
            "glove.n.02",
            "jewelry.n.01",
            "body_armor.n.01",
            "protective_garment.n.01",
            "spectacles.n.01",
        ],
    ),
    ("furniture", None, ["furniture.n.01"]),
    ("musical instrument", None, ["musical_instrument.n.01"]),
    (
        "electronics",
        None,
        [
            "electronic_equipment.n.01",
            "computer.n.01",
            "electronic_device.n.01",
            "camera.n.01",
            "receiver.n.01",
        ],
    ),
    ("household appliance", None, ["home_appliance.n.01", "kitchen_appliance.n.01"]),
    ("kitchenware", None, ["cookware.n.01", "kitchen_utensil.n.01"]),
    ("timepiece", None, ["timepiece.n.01"]),
    ("sports equipment", None, ["sports_equipment.n.01", "ball.n.01", "sports_implement.n.01"]),
    ("weapon", None, ["weapon.n.01"]),
    ("tool", None, ["tool.n.01", "hand_tool.n.01"]),
    ("container", None, ["container.n.01"]),
    ("object", None, ["artifact.n.01"]),
]

# Classes WordNet files somewhere unhelpful (keras class name -> topic id).
OVERRIDES = {
    "street_sign": "object",
    "traffic_light": "object",
    "red_wine": "drink",
    "espresso": "drink",
    "eggnog": "drink",
    "cup": "container",
    "bubble": "object",
    "sandbar": "landscape",
    "toilet_tissue": "object",
    "yurt": "building",
    "scuba_diver": "person",
    "groom": "person",
    "ballplayer": "person",
    # Found in review: WordNet files these as generic artifacts or coverings.
    "water_tower": "building and structure",
    "bell_cote": "building",
    "birdhouse": "building",
    "shoji": "building and structure",
    "window_screen": "building and structure",
    "window_shade": "building and structure",
    "pier": "building and structure",
    "drilling_platform": "building and structure",
    "radio_telescope": "building and structure",
    "solar_dish": "building and structure",
    "ski": "sports equipment",
    "puck": "sports equipment",
    "racket": "sports equipment",
    "harvester": "vehicle",
    "thresher": "vehicle",
    "cassette": "electronics",
    "television": "electronics",
    "loudspeaker": "electronics",
    "microphone": "electronics",
    "projector": "electronics",
    "photocopier": "electronics",
    "hard_disc": "electronics",
    "remote_control": "electronics",
    "cash_machine": "electronics",
    "broom": "tool",
    "paintbrush": "tool",
    "pick": "tool",
    "carpenter's_kit": "tool",
    "rule": "tool",
    "cloak": "clothing",
    "bib": "clothing",
    "pickelhaube": "clothing",
    "corn": "plant",
    "ear": "plant",
    "radio": "electronics",
    "stove": "household appliance",
    "space_heater": "household appliance",
    "electric_fan": "household appliance",
    "radiator": "household appliance",
    "Dutch_oven": "kitchenware",
    "strainer": "kitchenware",
    "gasmask": "clothing",
    "ski_mask": "clothing",
    "breastplate": "clothing",
    "parking_meter": "object",
    "Italian_greyhound": "dog",
    "whippet": "dog",
    "bullet_train": "rail vehicle",
    "freight_car": "rail vehicle",
    "minibus": "bus",
    "school_bus": "bus",
    "trolleybus": "bus",
    "motor_scooter": "motor vehicle",
    "tractor": "motor vehicle",
    "recreational_vehicle": "motor vehicle",
}


def closest_topic(synset: Synset, anchors: dict[str, str]) -> str | None:
    """Breadth-first up the hypernym graph; the nearest anchor wins.

    Ties (WordNet multiple inheritance, e.g. wheeled_vehicle -> vehicle AND container) go to
    the topic deeper in our tree, then to the one listed first in TOPICS.
    """
    order = {topic: i for i, (topic, _, _) in enumerate(TOPICS)}
    parents = {topic: parent for topic, parent, _ in TOPICS}

    def tree_depth(topic: str) -> int:
        depth = 0
        while parents[topic] is not None:
            topic, depth = parents[topic], depth + 1  # type: ignore[assignment]
        return depth

    frontier: deque[tuple[Synset, int]] = deque([(synset, 0)])
    seen: set[str] = set()
    best: tuple[int, int, int, str] | None = None
    while frontier:
        node, depth = frontier.popleft()
        if node.name() in seen or (best and depth > best[0]):
            continue
        seen.add(node.name())
        if node.name() in anchors:
            topic = anchors[node.name()]
            candidate = (depth, -tree_depth(topic), order[topic], topic)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        frontier.extend((h, depth + 1) for h in node.hypernyms() + node.instance_hypernyms())
    return best[3] if best else None


def main() -> None:
    nltk.download("wordnet", quiet=True)
    anchors = {anchor: topic for topic, _, names in TOPICS for anchor in names}
    index = json.loads(CLASS_INDEX.read_text())
    names = DISPLAY_NAMES.read_text(encoding="utf-8").splitlines()

    known_names = {keras_name for _, keras_name in index.values()}
    topic_ids = {topic for topic, _, _ in TOPICS}
    unknown = sorted(set(OVERRIDES) - known_names) + sorted(
        f"{name} -> {topic}" for name, topic in OVERRIDES.items() if topic not in topic_ids
    )
    if unknown:
        raise SystemExit(f"OVERRIDES reference unknown classes or topics: {unknown}")

    labels = []
    for i in range(len(index)):
        wnid, keras_name = index[str(i)]
        synset = wn.synset_from_pos_and_offset("n", int(wnid[1:]))
        topic = OVERRIDES.get(keras_name) or closest_topic(synset, anchors) or "object"
        labels.append({"id": wnid, "name": names[i], "topic": topic, "index": i})

    document = {
        "version": 1,
        "source": "ImageNet-1k classes placed under topics by WordNet hypernyms "
        "(scripts/build_taxonomy.py), then edited by hand",
        "topics": [{"id": t, "parent": p} for t, p, _ in TOPICS],
        "labels": labels,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write(document)

    counts = Counter(label["topic"] for label in labels)
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(labels)} labels")
    for topic, _, _ in TOPICS:
        print(f"  {counts.get(topic, 0):4}  {topic}")


def write(document: dict[str, object]) -> None:
    """One topic or label per line, so the file greps well and diffs show single labels."""

    def line(item: object) -> str:
        flow = yaml.safe_dump(item, default_flow_style=True, sort_keys=False, width=1_000_000)
        return "  - " + flow.strip()

    head = {k: v for k, v in document.items() if k not in ("topics", "labels")}
    header = yaml.safe_dump(head, sort_keys=False, width=100)
    topics = document["topics"]
    labels = document["labels"]
    if not isinstance(topics, list) or not isinstance(labels, list):
        raise SystemExit("document needs 'topics' and 'labels' lists")
    body = ["topics:", *map(line, topics), "labels:", *map(line, labels)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(header + "\n".join(body) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
