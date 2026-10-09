---
title: Image upload
slug: image-upload
status: beta
epic: E0
owner: vision
paths: ["src/synthwerk_vision/storage.py"]
api: ["POST /v1/images"]
events: []
mcp: []
updated: 2026-10-09
---

# Image upload

> **In 30 seconds**
> - The service stores one uploaded image on the local disk and returns a new id.
> - No endpoint reads the image back yet.

## What it does

- The service checks the image like a classification does: size, format and pixel count.
- The service names the file `<uuid4>.<ext>`. The extension comes from the decoded format.
- The service never uses the client file name.
- The service opens each file with `O_EXCL`, so an upload cannot overwrite a file.
- When a write fails, the service removes the partial file.

## How to use

1. Send the image as `multipart/form-data` in the field `image` to `POST /v1/images`.
2. Keep the `id` from the response.

## Rules and limits

| Rule | Value |
|---|---|
| Storage folder | `uploads` (`SYNTHWERK_VISION_UPLOAD_DIR`) |
| Image formats | JPEG, PNG, WebP |
| Maximum upload size | 5 MiB (`SYNTHWERK_VISION_MAX_UPLOAD_BYTES`) |
| Read back by id | Not available |

## Events

| Type | When |
|---|---|
| None. | |

## Errors

| Status | Problem type | Cause |
|---|---|---|
| 413 | None (FastAPI `detail`) | The image is larger than the upload limit or the pixel limit. |
| 415 | None (FastAPI `detail`) | The bytes are not a JPEG, PNG or WebP image. |
| 422 | None (FastAPI `detail`) | The request has no `image` field, or the file is empty. |

## Tests

- `tests/test_api.py` — `upload - valid image - stored under a generated id`
- `tests/test_api.py` — `upload - client file name - never reaches the file system`
- `tests/test_units.py` — `storage - write error - partial file removed`

## Changes

| Date | Change |
|---|---|
| 2026-10-09 | Rename to `synthwerk_vision`. The settings use the prefix `SYNTHWERK_VISION_`. |
