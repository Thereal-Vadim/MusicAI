"""MediaPipe Hand Landmarker model resolution (Tasks API)."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

log = logging.getLogger("musicai.mediapipe")

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_model_cache_path() -> Path:
    return _project_root() / "data" / "models" / MODEL_FILENAME


def resolve_hand_landmarker_model(explicit: Path | str | None = None) -> Path:
    """Return path to hand_landmarker.task, downloading to data/models on first use."""
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"MediaPipe hand model not found at {path}")

    cache = default_model_cache_path()
    if cache.exists() and cache.stat().st_size > 0:
        return cache

    cache.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading MediaPipe hand_landmarker.task → %s", cache)
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, cache)
    return cache
