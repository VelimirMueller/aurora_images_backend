# Aurora Images API

[![CI](https://github.com/VelimirMueller/aurora_images_backend/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/VelimirMueller/aurora_images_backend/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A FastAPI service that classifies any image into a label **and** its place in a topic tree
(`hut → building → building and structure`, `beagle → dog → mammal → animal`) and returns a
detailed JSON answer. It runs locally on CPU with ONNX Runtime and needs no GPU and no
external API. It is the backend for the Aurorae image frontend.

- **Safe uploads**: the client filename is never used, the type comes from the decoded image
  and not from `Content-Type`, and size and pixel limits are enforced while streaming.
- **Open vocabulary** (default backend): [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224)
  embeds the image and every label *text* into one space, so a label is just a string. The
  packaged taxonomy has 1068 labels: the 1000 ImageNet classes plus 68 everyday ones ImageNet
  lacks (hut, skyscraper, smartphone, sunset, aurora borealis, painting, screenshot …).
  Add a label by adding a line of YAML; nothing is retrained.
- **Topic roll-up**: a topic's score is the summed probability of every label below it. The
  model can be unsure *which* building (hut 0.32, boathouse 0.14, barn 0.11) but sure that it
  is a building (0.80).
- **Editable taxonomy**: generated from WordNet, committed as YAML; point
  `AURORA_TAXONOMY_PATH` at your own file.
- **Typed contract**: Pydantic response models drive the OpenAPI schema at `/docs`.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                                  # install runtime + dev dependencies from uv.lock
uv run python scripts/fetch_model.py     # both backends, pinned + SHA-256 verified (~1.5 GB)
                                         # or: fetch_model.py siglip | fetch_model.py mobilenet
uv run uvicorn aurora_images.main:create_app --factory --reload
```

Open http://localhost:8000/docs, or:

```bash
curl -F "image=@hut.jpg" http://localhost:8000/v1/classifications
```

The first start embeds all label texts once (about 9 s on an M-series CPU) and caches them in
`models/label_cache/`; later starts take about 0.5 s. Changing any label text invalidates the
cache automatically.

With Docker (weights are fetched and the label cache is built at image build time):

```bash
docker build -t aurora-images .                              # SigLIP (default)
docker build --build-arg BACKEND=mobilenet -t aurora-images:small .   # 14 MB of weights
docker run -p 8000:8000 -v aurora-uploads:/data/uploads aurora-images
```

## API

| Method | Path                  | Purpose                                     | Success | Errors        |
|--------|-----------------------|---------------------------------------------|---------|---------------|
| GET    | `/health`             | Liveness, and whether the model is loaded   | 200     |               |
| POST   | `/v1/images`          | Store an image, return its generated id     | 201     | 413, 415, 422 |
| POST   | `/v1/classifications` | Classify an image (not stored)              | 200     | 413, 415, 422, 503 |

Both POST endpoints take `multipart/form-data` with one field, `image` (JPEG, PNG or WebP).

Example: a photo of a wooden hut in a rice field (real output, `top_k=3`, ids shortened):

```json
{
  "request_id": "da71efe1…",
  "image": { "sha256": "99645f6c6ed4…", "content_type": "image/jpeg", "size_bytes": 149115, "width": 960, "height": 480 },
  "model": { "name": "siglip2-base-patch16-224", "taxonomy_version": 1 },
  "primary": {
    "id": "open:hut", "label": "hut", "score": 0.3229,
    "topic": "building", "path": ["building and structure", "building", "hut"]
  },
  "topic": { "id": "building and structure", "parent": null, "depth": 0, "score": 0.815 },
  "labels": [
    { "id": "open:hut", "label": "hut", "score": 0.3229, "topic": "building", "path": ["building and structure", "building", "hut"] },
    { "id": "n02859443", "label": "boathouse", "score": 0.1376, "topic": "building", "path": ["…"] },
    { "id": "n02793495", "label": "barn", "score": 0.1131, "topic": "building", "path": ["…"] }
  ],
  "topics": [
    { "id": "building and structure", "parent": null, "depth": 0, "score": 0.815 },
    { "id": "building", "parent": "building and structure", "depth": 1, "score": 0.7964 },
    { "id": "landscape", "parent": null, "depth": 0, "score": 0.1428 }
  ],
  "uncertain": false,
  "timings_ms": { "inference": 31.45, "rollup": 0.14, "total": 35.72 }
}
```

Spot checks on six Wikimedia Commons photos (2026-10-02); the real numbers are in
[Evaluation](#evaluation):

| Photo | `primary.path` (label score) | `topic` |
|---|---|---|
| Labrador | animal › mammal › dog › Labrador retriever (0.89) | animal 0.99 |
| Cat in snow | animal › mammal › cat › tabby (0.43) | animal 0.99 |
| Bicycle | vehicle › bicycle › bicycle-built-for-two (0.94, wrong leaf) | vehicle 0.98 |
| Wooden hut | building and structure › building › hut (0.32) | building and structure 0.82 |
| Skyscraper | building and structure › building › skyscraper (0.58) | building and structure 0.99 |
| Aurora (a painting) | landscape › sky › aurora borealis (0.57) | landscape 0.65 |

| Field | Meaning |
|---|---|
| `primary` | The most likely label, with the path from its root topic |
| `topic` | The most likely root ("meta") topic, such as `animal`, `vehicle` or `building and structure` |
| `labels` | Top-k labels (`AURORA_TOP_K`) |
| `topics` | Every topic scoring at least `AURORA_TOPIC_MIN_SCORE`, best first |
| `uncertain` | `true` when the root topic score is below `AURORA_UNCERTAIN_BELOW` |

Every response, errors included, carries an `X-Request-ID` header, which is also written to the
access log. A caller-supplied `X-Request-ID` is kept when it is 1–64 characters of
`[A-Za-z0-9_.-]`; otherwise the server generates one. Classification responses repeat it as
`request_id`.

## Taxonomy

Two packaged taxonomies, one per backend:

| File | Backend | Topics | Labels |
|---|---|---|---|
| [`taxonomy_open.yaml`](src/aurora_images/data/taxonomy_open.yaml) | `siglip` | 57 (22 roots) | 1068: extends the ImageNet file with 68 more |
| [`taxonomy.yaml`](src/aurora_images/data/taxonomy.yaml) | `mobilenet` | 53 (20 roots) | 1000, `index` = model output |

```yaml
topics:
  - {id: building, parent: building and structure}
labels:
  - {id: n03793489, name: mouse, topic: electronics, index: 673}
```

`taxonomy_open.yaml` does not repeat the 1000 ImageNet labels. It `extends`
`taxonomy.yaml` and adds `prompts`, 4 topics and 68 labels. Your own file can do the same,
which is the easiest way to add labels: a few lines on top of the packaged tree.

```yaml
# my-taxonomy.yaml, used with AURORA_TAXONOMY_PATH=my-taxonomy.yaml
extends: package:taxonomy_open.yaml
topics:
  - {id: aurora photo, parent: sky}
labels:
  - {id: my:corona, name: aurora corona, topic: aurora photo}
```

`extends` takes `package:<file>` or a path relative to the extending file, up to 4 levels.
Extending labels get their `index` automatically (after the base labels), and `prompts` may
only name labels that exist.

For SigLIP, a label is scored by the text `this is a photo of a {prompt or name}.`. The
`prompt` field disambiguates names a text model would misread ("mouse" is the computer
mouse). Every label text must be unique, or two labels get the same embedding.

`scripts/build_taxonomy.py` generates both files from WordNet: each ImageNet label goes
under the topic whose WordNet anchor is its *closest* hypernym. Its `TOPICS`, `OVERRIDES`,
`PROMPTS` and `OPEN_LABELS` tables are the source of truth: edit them and rerun. For your
own deployment you can also edit a copy of the YAML and set `AURORA_TAXONOMY_PATH`. On startup the service checks the tree
(unknown parents, cycles, duplicate ids, gaps in the indices, label count against the model's
outputs) and refuses to start if it is broken. This holds even when the model is missing: a
missing model is an ops state (503), while a broken taxonomy is a config bug.
If the model files are missing, the service still starts: `/health` reports
`"model_loaded": false` and classification returns 503.

### Backends

| `AURORA_BACKEND` | Model | Labels | Inference (CPU) | Weights |
|---|---|---|---|---|
| `siglip` (default) | SigLIP 2 base/16, 224 px, fp32 image and text towers | any text | ~30 ms | ~1.5 GB |
| `mobilenet` | MobileNetV2-12 | 1000 ImageNet classes | ~5 ms | 14 MB |

The build-time label cache covers the packaged taxonomy. With a custom
`AURORA_TAXONOMY_PATH`, the first container start embeds its labels (seconds) into
`AURORA_LABEL_CACHE_DIR`, so mount that directory as a volume to keep the cache across restarts.
The text tower runs only when label texts change (a cache miss): about 9 s for 1068 labels
with a ~3.2 GB memory peak (the fp32 Gemma vocabulary table alone is 786 MB), and then it is
dropped. Docker images ship with the cache built, so containers never load it unless you use a
custom taxonomy. In that case, give the first start about 4 GB of memory and mount the cache
directory. With the int8 text tower the image was 1.77 GB and the container was ready in
1.5 s using ~690 MB of RAM (2026-10-02). The fp32 text tower adds ~850 MB to the image; size
and RSS have not been re-measured since that change.

Both towers are **fp32 on purpose**:
- In a spike, the int8 image tower misread an aurora painting as a boat.
- The int8 text tower is accurate on ARM (cosine 0.996 to fp32) but not on x86 servers
  without VNNI (mean 0.959, min 0.872). On the x86 CI runner that dropped root-topic accuracy
  from 93.1 % to 83.0 %.
- fp32 gives identical embeddings on every architecture.

## Evaluation

[`eval/manifest.yaml`](eval/manifest.yaml) is a frozen set of **58 freely licensed Wikimedia
Commons photos** across 17 root topics. Every entry records the source page, author, license
and SHA-256. Images were picked by search query and license only, never by model output, then
checked by eye. Mixed images accept more than one topic (a girl hugging her dog: `person` or
`animal`). The metric is **root-topic top-1 accuracy**: is `topic` one of the accepted topics?

| Backend | Root-topic accuracy | 95 % interval (Wilson) | In top 3 | Latency/image |
|---|---|---|---|---|
| `siglip` | **93.1 %** (54/58) | 83.6–97.3 % | 98.3 % | 35 ms mean, 41 ms p95 |
| `mobilenet` | 70.7 % (41/58) | 58.0–80.8 % | 84.5 % | 7 ms mean, 10 ms p95 |

Measured 2026-10-02 on an M-series CPU. The intervals do not overlap, so the difference holds
even on 58 images. CI runs the eval on every PR and on pushes to `main`/`dev` and fails below 90 % for SigLIP. The
report is in the job summary.

```bash
uv run python scripts/evaluate.py                         # siglip
AURORA_BACKEND=mobilenet uv run python scripts/evaluate.py
```

What the misses show:
- **The catch-all bias of summed scores.** A sunflower became `object` although its top label
  was `daisy` (a plant). ImageNet has no sunflower, so probability spread across many weak
  labels, and the 113-label `object` topic collected more of it than `plant`. Broader label
  coverage fixes this better than any score trick. It is **not** patched here, because tuning
  the taxonomy on eval misses would make the eval meaningless.
- **Defensible disagreements**: a marina → `dock`, a waterfall with a dam → `dam`.
- MobileNet reads painted portraits as clothing and maps as jigsaw puzzles (1.00). Those are
  the limits of a closed 1000-class model.

Images are cached in `eval/images/` (in CI: an Actions cache keyed by the manifest). Only bytes
that match the manifest's SHA-256 are cached. If Commons re-renders or deletes a thumbnail, that
image is reported as *unavailable* instead of being scored, and the gate fails once more than
10 % are unavailable. The fix is then to rebuild the set: `scripts/build_eval_manifest.py
--force` followed by a new visual review, because Commons search order drifts between runs.

## Configuration

All settings are environment variables with the `AURORA_` prefix. See [`.env.example`](.env.example).

| Variable                  | Default                                        |
|---------------------------|------------------------------------------------|
| `AURORA_CORS_ORIGINS`     | `["http://localhost:5173","http://localhost:8080"]` |
| `AURORA_UPLOAD_DIR`       | `uploads`                                      |
| `AURORA_MAX_UPLOAD_BYTES` | `5242880` (5 MiB)                              |
| `AURORA_MAX_IMAGE_PIXELS` | `40000000`                                     |
| `AURORA_BACKEND`          | `siglip` (or `mobilenet`)                      |
| `AURORA_SIGLIP_DIR`       | `models/siglip2-base-patch16-224`              |
| `AURORA_LABEL_CACHE_DIR`  | `models/label_cache`                           |
| `AURORA_MODEL_PATH`       | `models/mobilenetv2-12.onnx` (mobilenet)       |
| `AURORA_TAXONOMY_PATH`    | unset (packaged taxonomy for the backend)      |
| `AURORA_ORT_THREADS`      | `0` (ONNX Runtime default; spin-waiting is off) |
| `AURORA_TOP_K`            | `5`                                            |
| `AURORA_TOPIC_MIN_SCORE`  | `0.05`                                         |
| `AURORA_UNCERTAIN_BELOW`  | `0.5`                                          |

## Development

```bash
uv run pytest            # tests + coverage gate (90 %)
uv run ruff check . && uv run ruff format --check .
uv run mypy              # strict
uv run pre-commit install
```

CI (`.github/workflows/ci.yml`) runs the same checks, the real-model test, a Docker build
and a container smoke test. Dependabot watches Python, Actions and Docker dependencies.

```
src/aurora_images/
  main.py         app factory, CORS, lifespan (loads storage + classifier)
  api/routes.py   HTTP layer and dependency wiring
  images.py       bounded reads and image validation
  storage.py      ImageStorage protocol + local disk implementation
  classifier.py   Classifier protocol (image -> probability vector) + MobileNetV2
  siglip.py       SigLIP 2: image/text towers, tokenizer, label-embedding cache
  taxonomy.py     topic tree, label paths, roll-up of probabilities into topic scores
  service.py      classifier + taxonomy -> labels, topics, uncertainty, timings
  evaluation.py   manifest loading, cached downloads, metrics, markdown/JSON reports
  data/taxonomy_open.yaml, data/taxonomy.yaml
  config.py       pydantic-settings
scripts/fetch_model.py      pinned model download
scripts/build_taxonomy.py   regenerate both taxonomies from WordNet (dev only)
scripts/warm_label_cache.py embed label texts ahead of time (Docker build)
scripts/evaluate.py         eval gate (root-topic accuracy, per-topic table, misses)
scripts/build_eval_manifest.py   rebuild the eval set from Commons (needs re-review)
```

## Model and data

| Item    | Source                                                                 | License      |
|---------|------------------------------------------------------------------------|--------------|
| SigLIP 2 weights + tokenizer | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224), ONNX export by [onnx-community](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX) (revision `ba1f3b0`) | Apache-2.0 |
| MobileNetV2 weights | [ONNX Model Zoo, MobileNetV2-12](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet) | Apache-2.0   |
| Labels  | [pytorch/hub `imagenet_classes.txt`](https://github.com/pytorch/hub)   | BSD-3-Clause |
| Synset ids | [Keras `imagenet_class_index.json`](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json) | Apache-2.0 |
| Eval images | 58 Wikimedia Commons files, per-file author + license in `eval/manifest.yaml` (CC BY / CC BY-SA / CC0 / public domain); not redistributed, downloaded on demand | per file |
| Topic tree | [WordNet 3.0](https://wordnet.princeton.edu/license-and-commercial-use) via NLTK | WordNet license (permissive) |

The weights were trained by their publishers (SigLIP 2 on WebLI, MobileNetV2 on
ImageNet-1k). This repository contains no
training data and no scraped images. Any future fine-tuning dataset will be documented
here with its source and license before it is used.

## Security

- Uploaded files are named `<uuid4>.<ext>`. The extension comes from the decoded format,
  and files open with `O_EXCL`, so an upload can never overwrite an existing file.
- CORS allows only the configured origins, without credentials.
- There is no authentication. Put the service behind a gateway, or add auth, before you
  expose it publicly.

## Roadmap

- Labels learned from example images at runtime, plus a feedback endpoint.
- An S3-compatible `ImageStorage`, plus a `GET /v1/images/{id}`.
- API keys and rate limiting.

## License

[MIT](LICENSE)
