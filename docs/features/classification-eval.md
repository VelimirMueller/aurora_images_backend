---
title: Classification eval gate
slug: classification-eval
status: stable
epic: E0
owner: vision
paths: ["src/synthwerk_vision/evaluation.py"]
api: []
events: []
mcp: []
updated: 2026-10-09
---

# Classification eval gate

> **In 30 seconds**
> - CI measures root-topic accuracy on 58 frozen Wikimedia Commons photos.
> - CI fails below 90 %. SigLIP 2 scores 93.1 % (54 of 58).

## What it does

- The manifest `eval/manifest.yaml` lists each photo with source, author, license and SHA-256.
- The harness downloads each photo once and keeps it in `eval/images/`.
- A download is accepted when the SHA-256 matches, or the dHash is within 8 bits.
- The harness counts a hit when `topic` is one of the accepted root topics.
- The report gives accuracy, a 95 % Wilson interval, top-3 accuracy and latency.

## How to use

1. Fetch the model: `uv run python scripts/fetch_model.py siglip`.
2. Run the gate: `uv run python scripts/evaluate.py --min-topic-accuracy 0.9`.
3. Read the report in `eval/reports/`.

## Rules and limits

| Rule | Value |
|---|---|
| Minimum root-topic accuracy | 90 % |
| Maximum unavailable photos | 10 % |
| Photos | 58, 17 root topics |
| SigLIP 2 result | 93.1 % (95 % interval 83.6–97.3 %) |

## Events

| Type | When |
|---|---|
| None. | |

## Errors

| Status | Problem type | Cause |
|---|---|---|
| None. | | |

## Tests

- `tests/test_evaluation.py` — `evaluate - cached images - scored without network`
- `tests/test_evaluation.py` — `download - re-encoded same picture - scored and reported`
- `tests/test_evaluation.py` — `download - different picture with a dHash - rejected`

## Changes

| Date | Change |
|---|---|
| 2026-10-09 | Rename to `synthwerk_vision`. The eval User-Agent names `synthwerk-vision`. |
