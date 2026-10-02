# Aurora Images API

[![CI](https://github.com/VelimirMueller/aurora_images_backend/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/VelimirMueller/aurora_images_backend/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A small FastAPI service that takes image uploads and classifies them with a pretrained
CNN on ONNX Runtime. It is the backend for the Aurorae image frontend.

- **Safe uploads**: the client filename is never used, the type comes from the decoded image
  and not from `Content-Type`, and size and pixel limits are enforced while streaming.
- **Classification**: MobileNetV2 (ImageNet, 1000 classes) returns the top-k labels with
  probabilities. The model sits behind a `Classifier` protocol, so a fine-tuned model can
  replace it without API changes.
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
curl -F "image=@cat.jpg" http://localhost:8000/v1/classifications
# {"model":"mobilenetv2-12","predictions":[{"label":"<imagenet class>","score":<0..1>}, ...]}
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
| `AURORA_LABELS_PATH`      | `models/imagenet_classes.txt`                  |
| `AURORA_TOP_K`            | `5`                                            |

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
  classifier.py   Classifier protocol + ONNX Runtime implementation
  config.py       pydantic-settings
scripts/fetch_model.py   pinned model download
```

## Model and data

| Item    | Source                                                                 | License      |
|---------|------------------------------------------------------------------------|--------------|
| Weights | [ONNX Model Zoo, MobileNetV2-12](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet) | Apache-2.0   |
| Labels  | [pytorch/hub `imagenet_classes.txt`](https://github.com/pytorch/hub)   | BSD-3-Clause |

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

- Fine-tune on a licensed, documented aurora/sky dataset and publish an evaluation report.
- An S3-compatible `ImageStorage`, plus a `GET /v1/images/{id}`.
- API keys and rate limiting.

## License

[MIT](LICENSE)
