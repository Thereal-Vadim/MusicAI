# Architecture

## Monorepo layout

```
apps/
  api/          FastAPI + SQLite
  worker/       ML pipeline orchestration
  web/          Next.js UI
packages/
  inference/    Model registry + adapters
  judge/        Music Theory Judge
  tab_schema/   TabDocument Pydantic models
benchmarks/     Manifest + spike runner
tests/          Unit + integration tests
```

## Inference registry

All neural models are accessed through `ModelRegistry` with adapters:

- `demucs/htdemucs_6s`
- `basic-pitch/v1`
- `mediapipe/hands`
- `librosa/beat`

Switch runtime via `INFERENCE_RUNTIME=local|cloud` (cloud stubs only in MVP).

## Judge

Deterministic post-processing after fusion. Never calls external LLM APIs.

Rules: key detection, scale/chord membership, snap-to-grid for low-confidence outliers, playability checks.

## Data contract

`packages/tab-schema/tab.schema.json` is the canonical TabDocument format shared by worker, API, and web.
