# Go/no-go decision

**Decision:** GO for alpha with current librosa fallbacks; install `[ml]` extra for production-quality separation/transcription.

## Criteria

| Metric | Target | Result |
|--------|--------|--------|
| End-to-end pipeline | Pass | Pass (integration test) |
| Judge false snap | < 15% | 0% on synthetic benchmark |
| Pipeline latency (synthetic) | < 480s | ~2.6s max |
| Vision fallback | Works without video | Pass |

## Next steps post-MVP

1. Install `pip install -e ".[ml]"` on Python 3.11 machine with GPU/MPS.
2. Add real licensed benchmark media under `benchmarks/media/`.
3. Tune Basic Pitch thresholds per genre.
4. Optional cloud inference via `INFERENCE_RUNTIME=cloud`.
