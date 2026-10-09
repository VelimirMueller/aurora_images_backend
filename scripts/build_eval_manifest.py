"""Build eval/manifest.yaml from Wikimedia Commons: licensed photos per expected root topic.

Images are chosen by search query and license only, never by what a model predicts. The first
results per query with a free license are taken. Each entry was then checked by eye on a
contact sheet; REVIEW below holds the outcome (drop, or extra acceptable topics).

The committed eval/manifest.yaml is the frozen eval set. Commons search order drifts between
runs (a rebuild on 2026-10-02 already swapped one museum painting), so a rebuild yields a
*different* set that must be reviewed by eye again; that is why --force is required.
Images are not committed; scripts/evaluate.py downloads them and verifies the SHA-256.

Usage: uv run python scripts/build_eval_manifest.py --force
"""

import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

from synthwerk_vision.evaluation import USER_AGENT
from synthwerk_vision.images import dhash

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "eval" / "manifest.yaml"
IMAGES = ROOT / "eval" / "images"
API = "https://commons.wikimedia.org/w/api.php"
PER_QUERY = 2
FREE_LICENSE = re.compile(r"^(CC BY(-SA)? [0-9.]+|CC0( 1\.0)?|Public domain)$", re.IGNORECASE)

# (search query, expected root topic). Two images per query.
QUERIES: list[tuple[str, str]] = [
    ("dog portrait", "animal"),
    ("domestic cat", "animal"),
    ("cow pasture", "animal"),
    ("parrot", "animal"),
    ("butterfly on flower", "animal"),
    ("portrait photograph of a man", "person"),
    ("group of people", "person"),
    ("oak tree", "plant"),
    ("sunflower", "plant"),
    ("pizza", "food and drink"),
    ("salad bowl", "food and drink"),
    ("car on street", "vehicle"),
    ("bicycle", "vehicle"),
    ("airliner", "vehicle"),
    ("passenger train", "vehicle"),
    ("sailing boat", "vehicle"),
    ("house exterior", "building and structure"),
    ("church building", "building and structure"),
    ("bridge river", "building and structure"),
    ("mountain hut", "building and structure"),
    ("mountain landscape", "landscape"),
    ("sandy beach", "landscape"),
    ("waterfall", "landscape"),
    ("laptop computer", "electronics"),
    ("smartphone", "electronics"),
    ("armchair", "furniture"),
    ("sneakers shoes", "clothing"),
    ("acoustic guitar", "musical instrument"),
    ("oil painting museum", "art"),
    ("map of europe", "document"),
    ("football ball", "sports equipment"),
    ("refrigerator", "household appliance"),
    ("hammer tool", "tool"),
    ("frying pan", "kitchenware"),
]

# Outcome of the visual check (2026-10-02), keyed by Commons file title.
# "drop": not a fair example of the query; a list: further acceptable root topics.
REVIEW: dict[str, str | list[str]] = {
    "File:Human\u2013canine friendship - girl hugging her dog tightly at golden hour in Laos.jpg": [
        "person"
    ],  # dog portrait
    "File:A convenience by Glapthorn Cow Pasture - geograph.org.uk - 3180568.jpg": "drop",  # cow pasture
    "File:Asclepias syriaca close up of the seed copy.jpg": "drop",  # butterfly on flower
    "File:Portrait of a Man, Said to be Christopher Columbus.jpg": [
        "art"
    ],  # portrait photograph of a man
    "File:Antonello da Messina - Portrait of a Man - National Gallery London.jpg": [
        "art"
    ],  # portrait photograph of a man
    "File:Deelerwoud, 09-05-2024 (d.j.b.) 16.jpg": ["landscape"],  # oak tree
    "File:Eik met uitgebroken kroon. Locatie, Kroondomein Het Loo. 25-12-2020 (d.j.b.) 02.jpg": [
        "landscape"
    ],  # oak tree
    "File:Salad Bowl MET DP258642.jpg": "drop",  # salad bowl
    "File:Salad bowl, Chantilly Porcelain Factory, c. 1735-1740, soft-paste porcelain - Wadsworth Atheneum - Hartford, CT - DSC05417.jpg": "drop",  # salad bowl
    "File:Car radio antenna on Mazda 323 compacted in Tuntorp.jpg": "drop",  # car on street
    "File:Sleeping, homeless children - Jacob Riis.jpg": "drop",  # car on street
    "File:Parked bicycle with graffitied building facade and doors in Amsterdam.jpg": [
        "art"
    ],  # bicycle
    "File:Bicycle reflections.jpg": ["person"],  # bicycle
    "File:Airliner Contrail.jpg": "drop",  # airliner
    "File:Z\u00f6bigker Hafen, Cospudener See, Markkleeberg, 1709102012, ako.jpg": [
        "landscape"
    ],  # sailing boat
    "File:Zoutelande (NL), Strand, Blick auf die Nordsee -- 2022 -- 4984.jpg": [
        "landscape"
    ],  # sailing boat
    "File:Old house typical of the island of Margarita.jpg": ["furniture"],  # house exterior
    "File:New River Gorge Bridge.jpg": ["landscape"],  # bridge river
    "File:Lukmanierpass, Passo del Lucomagno. 20-09-2022. (actm.) 26.jpg": [
        "landscape"
    ],  # mountain hut
    "File:Glecksteinh\u00fctte front 20200121.jpg": ["landscape"],  # mountain hut
    "File:Paul Bril 002.jpg": ["art"],  # mountain landscape
    "File:Careening of a pirogue on a sandy beach of a tiny island in the Mekong near Don Det in Laos.jpg": [
        "vehicle"
    ],  # sandy beach
    "File:Phu Sang Waterfall 01.jpg": ["person"],  # waterfall
    "File:Smartphone ownership in 2013.jpg": "drop",  # smartphone
    "File:Mona Lisa, by Leonardo da Vinci, from C2RMF retouched.jpg": "drop",  # armchair
    "File:Lobby lounge of Amantaka Suite Amantaka luxury Resort & Hotel Luang Prabang Laos.jpg": [
        "building and structure"
    ],  # armchair
    "File:Vans sneakers and socks.jpg": ["person"],  # sneakers shoes
    "File:Harri Stojka 30.08.2008c.jpg": ["person"],  # acoustic guitar
    "File:Albert Bierstadt - Among the Sierra Nevada, California - Google Art Project.jpg": [
        "landscape"
    ],  # oil painting museum
    "File:Edams Museum (1530) - First Floor - 19th century Oil Painting depicting the Edam Cheese Market.jpg": [
        "building and structure"
    ],  # oil painting museum
    "File:Map of Europe, 1946 (25289557032).jpg": ["art"],  # map of europe
    "File:Jakarta old football.jpg": ["person"],  # football ball
    "File:Alg\u00e9rie - Arm\u00e9nie - 20140531 - Yacine Brahimi (Alg) face \u00e0 Taron Voskanyan (Arm).jpg": [
        "person"
    ],  # football ball
    "File:Open refrigerator with food at night.jpg": ["food and drink"],  # refrigerator
    "File:Armorer's Hammer MET 12.230.2 001feb2014.jpg": ["weapon"],  # hammer tool
    "File:Frying pan Naxos NAMA20935.jpg": "drop",  # frying pan
}


def api(params: dict[str, str]) -> dict[str, object]:
    url = API + "?" + urllib.parse.urlencode({"format": "json", **params})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed https host
        result: dict[str, object] = json.load(response)
        return result


def candidates(query: str) -> list[dict[str, object]]:
    data = api(
        {
            "action": "query",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrlimit": "20",
            "gsrsearch": f"{query} filetype:bitmap",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size",
            "iiurlwidth": "640",
            "iiextmetadatafilter": "LicenseShortName|Artist",
        }
    )
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p["index"])  # type: ignore[union-attr]
    found = []
    for page in pages:
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "")
        if (
            info["mime"] != "image/jpeg"
            or info["width"] < 480
            or not FREE_LICENSE.match(license_name)
        ):
            continue
        artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
        found.append(
            {
                "title": page["title"],
                "page": info["descriptionurl"],
                "url": info["thumburl"],
                "license": license_name,
                "author": artist or "unknown",
            }
        )
        if len(found) == PER_QUERY:
            break
    return found


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https host
        data: bytes = response.read()
        return data


def main() -> None:
    if MANIFEST.exists() and "--force" not in sys.argv:
        raise SystemExit(
            f"{MANIFEST} is the frozen eval set; pass --force to rebuild and re-review"
        )
    IMAGES.mkdir(parents=True, exist_ok=True)
    entries = []
    seen: set[str] = set()
    for query, topic in QUERIES:
        found = candidates(query)
        if len(found) < PER_QUERY:
            print(
                f"WARNING: {query!r} has only {len(found)} free-licensed candidates",
                file=sys.stderr,
            )
        for item in found:
            title = str(item["title"])
            review = REVIEW.get(title)
            if title in seen or review == "drop":
                continue
            seen.add(title)
            data = download(str(item["url"]))
            digest = hashlib.sha256(data).hexdigest()
            (IMAGES / f"{digest[:16]}.jpg").write_bytes(data)
            entries.append(
                {
                    **item,
                    "sha256": digest,
                    "dhash": dhash(data),
                    "query": query,
                    "topics": [topic, *(review if isinstance(review, list) else [])],
                }
            )
            time.sleep(0.5)  # be polite to the Commons API
    MANIFEST.write_text(
        yaml.safe_dump(
            {
                "source": "Wikimedia Commons, free licenses only (scripts/build_eval_manifest.py)",
                "images": entries,
            },
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )
    )
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(entries)} images")


if __name__ == "__main__":
    main()
