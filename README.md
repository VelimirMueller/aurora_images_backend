# Aurora Images API

[![CI](https://github.com/VelimirMueller/aurora_images_backend/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/VelimirMueller/aurora_images_backend/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A small FastAPI service that takes image uploads and classifies them with a pretrained
CNN on ONNX Runtime. It is the backend for the Aurorae image frontend.

- **Safe uploads**: the client filename is never used, the type comes from the decoded image
  and not from `Content-Type`, and size and pixel limits are enforced while streaming.
- **Hierarchical classification**: MobileNetV2 (ImageNet, 1000 classes) predicts the label.
  A topic tree places it in context (`animal → mammal → dog → Labrador retriever`), and topic
  scores add up the probability of every label below a topic. The model can be unsure *which*
  dog (0.53) but sure that it is a dog (0.97) and an animal (0.99).
- **Editable taxonomy**: the tree is a YAML file generated from WordNet. Edit it by hand, or
  point `AURORA_TAXONOMY_PATH` at your own file.
- **Typed contract**: Pydantic response models drive the OpenAPI schema at `/docs`.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                                  # install runtime + dev dependencies from uv.lock
uv run python scripts/fetch_model.py     # download weights + labels (SHA-256 verified, ~14 MB)
uv run uvicorn aurora_images.main:create_app --factory --reload
```

Open http://localhost:8000/docs, or:

```bash
curl -F "image=@bike.jpg" http://localhost:8000/v1/classifications
```

With Docker (the model is fetched at build time):

```bash
docker build -t aurora-images .
docker run -p 8000:8000 -v aurora-uploads:/data/uploads aurora-images
```

## API

| Method | Path                  | Purpose                                     | Success | Errors        |
|--------|-----------------------|---------------------------------------------|---------|---------------|
| GET    | `/health`             | Liveness, and whether the model is loaded   | 200     |               |
| POST   | `/v1/images`          | Store an image, return its generated id     | 201     | 413, 415, 422 |
| POST   | `/v1/classifications` | Classify an image (not stored)              | 200     | 413, 415, 422, 503 |

Both POST endpoints take `multipart/form-data` with one field, `image` (JPEG, PNG or WebP).

Example classification of a photo of a bicycle (real output, `labels` shortened):

```json
{
  "request_id": "9b0c…",
  "image": { "sha256": "…", "content_type": "image/jpeg", "size_bytes": 120927, "width": 960, "height": 640 },
  "model": { "name": "mobilenetv2-12", "taxonomy_version": 1 },
  "primary": {
    "id": "n02835271", "label": "bicycle-built-for-two", "score": 0.628,
    "topic": "bicycle", "path": ["vehicle", "bicycle", "bicycle-built-for-two"]
  },
  "topic": { "id": "vehicle", "parent": null, "depth": 0, "score": 0.9964 },
  "labels": [ { "id": "n02835271", "label": "bicycle-built-for-two", "score": 0.628, "topic": "bicycle", "path": ["…"] } ],
  "topics": [
    { "id": "vehicle", "parent": null, "depth": 0, "score": 0.9964 },
    { "id": "bicycle", "parent": "vehicle", "depth": 1, "score": 0.8147 }
  ],
  "uncertain": false,
  "timings_ms": { "inference": 4.94, "rollup": 0.11, "total": 5.86 }
}
```

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

[`src/aurora_images/data/taxonomy.yaml`](src/aurora_images/data/taxonomy.yaml) has 53 topics
in a tree (20 roots) and the 1000 ImageNet labels, each pinned to a model output index and
filed under its most specific topic:

```yaml
topics:
  - {id: vehicle, parent: null}
  - {id: bicycle, parent: vehicle}
labels:
  - {id: n03792782, name: mountain bike, topic: bicycle, index: 671}
```

`scripts/build_taxonomy.py` generated it from WordNet. Each label goes under the topic whose
WordNet anchor is its *closest* hypernym, and a short override list fixes WordNet quirks
(such as `yurt`). After that the file is hand-edited data: move a label by changing its
`topic`, or add a topic with a `parent`. On startup the service checks the tree
(unknown parents, cycles, duplicate ids, gaps in the indices, label count against the model's
outputs) and refuses to start if it is broken. This holds even when the model is missing: a
missing model is an ops state (503), while a broken taxonomy is a config bug.
If the model files are missing, the service still starts: `/health` reports
`"model_loaded": false` and classification returns 503.

## Configuration

All settings are environment variables with the `AURORA_` prefix. See [`.env.example`](.env.example).

| Variable                  | Default                                        |
|---------------------------|------------------------------------------------|
| `AURORA_CORS_ORIGINS`     | `["http://localhost:5173","http://localhost:8080"]` |
| `AURORA_UPLOAD_DIR`       | `uploads`                                      |
| `AURORA_MAX_UPLOAD_BYTES` | `5242880` (5 MiB)                              |
| `AURORA_MAX_IMAGE_PIXELS` | `40000000`                                     |
| `AURORA_MODEL_PATH`       | `models/mobilenetv2-12.onnx`                   |
| `AURORA_TAXONOMY_PATH`    | unset (uses the packaged taxonomy)             |
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
  classifier.py   Classifier protocol (image -> probability vector) + ONNX Runtime implementation
  taxonomy.py     topic tree, label paths, roll-up of probabilities into topic scores
  service.py      classifier + taxonomy -> labels, topics, uncertainty, timings
  data/taxonomy.yaml
  config.py       pydantic-settings
scripts/fetch_model.py      pinned model download
scripts/build_taxonomy.py   regenerate the taxonomy from WordNet (dev only)
```

## Model and data

| Item    | Source                                                                 | License      |
|---------|------------------------------------------------------------------------|--------------|
| Weights | [ONNX Model Zoo, MobileNetV2-12](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet) | Apache-2.0   |
| Labels  | [pytorch/hub `imagenet_classes.txt`](https://github.com/pytorch/hub)   | BSD-3-Clause |
| Synset ids | [Keras `imagenet_class_index.json`](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json) | Apache-2.0 |
| Topic tree | [WordNet 3.0](https://wordnet.princeton.edu/license-and-commercial-use) via NLTK | WordNet license (permissive) |

The weights were trained by their publishers on ImageNet-1k. This repository contains no
training data and no scraped images. Any future fine-tuning dataset will be documented
here with its source and license before it is used.

## Security

- Uploaded files are named `<uuid4>.<ext>`. The extension comes from the decoded format,
  and files open with `O_EXCL`, so an upload can never overwrite an existing file.
- CORS allows only the configured origins, without credentials.
- There is no authentication. Put the service behind a gateway, or add auth, before you
  expose it publicly.

## Roadmap

- Open-vocabulary labels with SigLIP 2, so labels outside ImageNet (such as "hut") work.
- Labels learned from example images at runtime, plus a feedback endpoint.
- An evaluation script with per-topic accuracy as a CI gate.
- An S3-compatible `ImageStorage`, plus a `GET /v1/images/{id}`.
- API keys and rate limiting.

## License

[MIT](LICENSE)
