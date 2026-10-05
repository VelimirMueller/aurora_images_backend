"""Generate the packaged taxonomies from WordNet.

- taxonomy.yaml: the 1000 ImageNet-1k classes, indexed like the MobileNetV2 outputs.
- taxonomy_open.yaml: the same classes with text prompts, plus open-vocabulary labels
  ImageNet lacks (hut, skyscraper, smartphone, sunset ...). SigLIP 2 scores these by text.

Every ImageNet class is a WordNet synset. Each class goes under the topic whose WordNet
anchor is the *closest* hypernym (breadth-first over all hypernym paths). TOPICS, OVERRIDES,
PROMPTS and OPEN_LABELS below are the source of truth: edit them and rerun.

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
DATA = ROOT / "src" / "aurora_images" / "data"

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


# Text prompts by WordNet id, for ImageNet names that are duplicated or ambiguous as text
# ("mouse" is the animal to a text model, the ImageNet class is the computer mouse).
# They go to taxonomy_open.yaml as `prompts`. Only for duplicated display names (crane,
# maillot) does the prompt also replace the name in taxonomy.yaml, so names stay unique.
PROMPTS = {
    "n02012849": "crane (bird)",
    "n03126707": "construction crane",
    "n03710637": "maillot (dance tights)",
    "n03710721": "one-piece swimsuit",
    "n03594734": "jeans",
    "n13133613": "ear of corn",
    "n12768682": "rose hip",
    "n03832673": "notebook computer",
    "n06359193": "website screenshot",
    "n07579787": "plate of food",
    "n01871265": "elephant with tusks",
    "n03782006": "computer monitor",
    "n04152593": "computer screen",
    "n03793489": "computer mouse",
    "n03337140": "filing cabinet",
    "n03584829": "clothes iron",
    "n04554684": "washing machine",
    "n03930313": "pickaxe",
    "n04118776": "measuring ruler",
    "n04372370": "light switch",
    "n04243546": "slot machine",
    "n04254680": "cotton swab",
    "n01847000": "drake (male duck)",
    "n02389026": "sorrel horse",
    "n02412080": "ram (sheep)",
    "n02395406": "hog (pig)",
    "n02403003": "ox",
    "n01514859": "hen (chicken)",
    "n01514668": "rooster",
    "n01440764": "tench (fish)",
    "n03595614": "jersey (shirt)",
    "n04325704": "stole (scarf)",
    "n03014705": "chest (wooden box)",
    "n04125021": "safe (strongbox)",
}

# Extra topics and labels only the open-vocabulary backend can score.
OPEN_TOPICS: list[tuple[str, str | None]] = [
    ("art", None),
    ("document", None),
    ("sky", "landscape"),
    ("city", "building and structure"),
]
OPEN_LABELS: list[tuple[str, str]] = [
    ("hut", "building"),
    ("house", "building"),
    ("wooden cabin", "building"),
    ("farmhouse", "building"),
    ("apartment building", "building"),
    ("office building", "building"),
    ("skyscraper", "building"),
    ("lighthouse", "building"),
    ("tower", "building and structure"),
    ("city skyline", "city"),
    ("street", "city"),
    ("road", "building and structure"),
    ("man", "person"),
    ("woman", "person"),
    ("child", "person"),
    ("baby", "person"),
    ("group of people", "person"),
    ("crowd", "person"),
    ("horse", "hoofed mammal"),
    ("cow", "hoofed mammal"),
    ("sheep", "hoofed mammal"),
    ("deer", "hoofed mammal"),
    ("rabbit", "mammal"),
    ("chicken", "bird"),
    ("duck", "bird"),
    ("motorcycle", "motor vehicle"),
    ("airplane", "aircraft"),
    ("helicopter", "aircraft"),
    ("train", "rail vehicle"),
    ("tram", "rail vehicle"),
    ("electric scooter", "vehicle"),
    ("smartphone", "electronics"),
    ("tablet computer", "electronics"),
    ("headphones", "electronics"),
    ("smartwatch", "electronics"),
    ("tree", "plant"),
    ("bouquet of flowers", "flower"),
    ("grass", "plant"),
    ("cactus", "plant"),
    ("houseplant", "plant"),
    ("forest", "landscape"),
    ("beach", "landscape"),
    ("mountain", "landscape"),
    ("river", "landscape"),
    ("lake", "landscape"),
    ("waterfall", "landscape"),
    ("desert", "landscape"),
    ("snowy landscape", "landscape"),
    ("field", "landscape"),
    ("sunset", "sky"),
    ("night sky with stars", "sky"),
    ("aurora borealis", "sky"),
    ("cloudy sky", "sky"),
    ("burger", "dish"),
    ("salad", "dish"),
    ("cake", "food and drink"),
    ("sushi", "dish"),
    ("coffee", "drink"),
    ("painting", "art"),
    ("drawing", "art"),
    ("sculpture", "art"),
    ("graffiti", "art"),
    ("text document", "document"),
    ("screenshot", "document"),
    ("chart", "document"),
    ("map", "document"),
    ("logo", "document"),
    ("handwriting", "document"),
]


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

    # Display names used by more than one class (crane, maillot) get their PROMPTS text as the
    # name in both files, so every label name is unique.
    duplicated = {name for name, count in Counter(names).items() if count > 1}
    labels = []
    for i in range(len(index)):
        wnid, keras_name = index[str(i)]
        synset = wn.synset_from_pos_and_offset("n", int(wnid[1:]))
        topic = OVERRIDES.get(keras_name) or closest_topic(synset, anchors) or "object"
        if names[i] in duplicated and wnid not in PROMPTS:
            raise SystemExit(f"{wnid} shares the name {names[i]!r}; add it to PROMPTS")
        name = PROMPTS[wnid] if names[i] in duplicated else names[i]
        labels.append({"id": wnid, "name": name, "topic": topic, "index": i})
    if unknown_prompts := sorted(set(PROMPTS) - {label["id"] for label in labels}):
        raise SystemExit(f"PROMPTS reference unknown label ids: {unknown_prompts}")

    topics = [{"id": t, "parent": p} for t, p, _ in TOPICS]
    write(
        "taxonomy.yaml",
        "ImageNet-1k classes placed under topics by WordNet hypernyms (scripts/build_taxonomy.py)",
        topics,
        labels,
    )

    # The open taxonomy only adds to taxonomy.yaml; Taxonomy.load merges the two.
    extra_topics = [{"id": t, "parent": p} for t, p in OPEN_TOPICS]
    topic_ids = {t["id"] for t in topics} | {t["id"] for t in extra_topics}
    taken = {str(label["name"]) for label in labels} | topic_ids
    extra_labels = []
    for name, topic in OPEN_LABELS:
        if topic not in topic_ids or name in taken:
            raise SystemExit(
                f"open label {name!r}: unknown topic or name clashes with a label/topic"
            )
        taken.add(name)
        extra_labels.append({"id": "open:" + name.replace(" ", "-"), "name": name, "topic": topic})
    write(
        "taxonomy_open.yaml",
        "taxonomy.yaml plus text prompts and open-vocabulary labels, scored by text "
        "(scripts/build_taxonomy.py)",
        extra_topics,
        extra_labels,
        extends="package:taxonomy.yaml",
        prompts=PROMPTS,
    )


def write(
    filename: str,
    source: str,
    topics: list[dict[str, object]],
    labels: list[dict[str, object]],
    *,
    extends: str | None = None,
    prompts: dict[str, str] | None = None,
) -> None:
    """One topic or label per line, so the file greps well and diffs show single labels."""

    def line(item: dict[str, object]) -> str:
        flow = yaml.safe_dump(item, default_flow_style=True, sort_keys=False, width=1_000_000)
        return "  - " + flow.strip()

    head: dict[str, object] = {"version": 1, "source": source}
    if extends:
        head["extends"] = extends
    header = yaml.safe_dump(head, sort_keys=False, width=100)
    body = []
    if prompts:
        body += [
            "prompts:",
            *(f"  {key}: {json.dumps(value)}" for key, value in prompts.items()),
        ]
    body += ["topics:", *map(line, topics), "labels:", *map(line, labels)]
    path = DATA / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(body) + "\n", encoding="utf-8")
    counts = Counter(str(label["topic"]) for label in labels)
    print(f"wrote {path.relative_to(ROOT)}: {len(topics)} topics, {len(labels)} labels")
    for topic in topics:
        print(f"  {counts.get(str(topic['id']), 0):4}  {topic['id']}")


if __name__ == "__main__":
    main()
