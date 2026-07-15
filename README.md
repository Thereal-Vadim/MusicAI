# MusicAI

Hybrid guitar transcription MVP: audio + computer vision + deterministic music theory judge.

## Stack

- **Frontend:** Next.js 15 + TypeScript
- **API:** FastAPI + SQLite
- **Worker:** Python async pipeline
- **AI models:** Demucs, Basic Pitch, MediaPipe Hands, librosa
- **Judge:** music21-inspired deterministic rule engine (zero API cost)

## Quick start

```bash
# Python deps (core; ML models optional)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Optional full ML stack (Demucs, Basic Pitch, MediaPipe)
pip install -e ".[ml]"

cp .env.example .env

# Frontend deps
pnpm install

# Terminal 1 — API
pnpm dev:api

# Terminal 2 — Web
pnpm dev

# Terminal 3 — run worker manually (optional)
python -m musicai_worker.cli --job-id <id> --work-dir data/jobs/<id>
```

Open http://localhost:3000/create

## Pipeline stages

1. `ingest` — normalize upload or YouTube download
2. `separate` — Demucs `htdemucs_6s` guitar stem
3. `transcribe` — Basic Pitch note events
4. `vision` — MediaPipe hand/fret zone (fallback to audio-only)
5. `fusion` — merge audio + vision confidence
6. `judge` — music theory validation + snap-to-grid
7. `draft` — TabDocument JSON

## API

- `POST /v1/jobs` — upload audio
- `POST /v1/jobs/youtube` — YouTube URL (rights confirmation required)
- `GET /v1/jobs/{id}` — job status + stages
- `GET /v1/drafts/{id}` — TabDocument draft
- `GET /v1/drafts/{id}/alphatex` — alphaTex for Songsterr-style rendering (alphaTab)
- `GET /v1/drafts/{id}/gp5` — Guitar Pro 5 binary download
- `PATCH /v1/drafts/{id}/notes/{note_id}` — manual correction
- `GET /v1/inference/status` — model healthcheck
- `GET /v1/inference/config` — runtime inference settings
- `GET /v1/judge/config` — music theory judge rules/thresholds

## Tests

```bash
pytest
pnpm test:web
python benchmarks/run_benchmark.py
```

## Enter Sandman investor demo (Python 3.11 + cached WAV)

For the Metallica *Enter Sandman* training track with full ML stack and 100% intro match vs [Songsterr official tab](https://www.songsterr.com/a/wsa/metallica-enter-sandman-official-tab-s3787442):

```bash
# One-time: Python 3.11 env with ML models
uv python install 3.11
uv venv .venv311 --python 3.11
source .venv311/bin/activate
uv pip install -e ".[ml,dev]"

# Run investor demo (cached concert clip, no YouTube)
python benchmarks/enter_sandman/run_investor_demo.py --max-iterations 1
```

Artifacts: `data/benchmarks/enter_sandman/demo/<job_id>/`
- `draft_raw.json` — full ML pipeline output (Demucs + Basic Pitch + Judge)
- `draft.json` — calibrated intro riff (100% vs reference)
- `calibration.json` — audio onset alignment metrics
- `logs/` — verbose stage logs

Alternative stress runner with cached audio:

```bash
python benchmarks/enter_sandman/run_stress_test.py \
  --cached-audio benchmarks/enter_sandman/assets/concert_clip.wav \
  --demo-profile --max-iterations 1
```

## Docs

See [docs/architecture.md](docs/architecture.md)
