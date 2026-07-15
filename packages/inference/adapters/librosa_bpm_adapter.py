"""Librosa BPM adapter."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from inference.adapters.base import BaseModelAdapter
from inference.schemas.model_io import BpmInput, BpmOutput

log = logging.getLogger("musicai.librosa_bpm")

DEFAULT_BPM = 120.0


def scalar_tempo(tempo: object) -> float:
    """Normalize librosa tempo output (scalar or 0-d/1-d ndarray) to float BPM."""
    arr = np.asarray(tempo, dtype=float).reshape(-1)
    if arr.size == 0:
        return DEFAULT_BPM
    return float(arr[0])


class LibrosaBpmAdapter(BaseModelAdapter):
    def __init__(self, model_id: str = "librosa/beat") -> None:
        self.model_id = model_id
        self.runtime = "local"

    def healthcheck(self) -> bool:
        try:
            import librosa  # noqa: F401

            return True
        except ImportError:
            return False

    async def predict(self, input_data: BpmInput) -> BpmOutput:
        return await asyncio.to_thread(self._detect, input_data)

    def _detect(self, input_data: BpmInput) -> BpmOutput:
        try:
            import librosa

            y, sr = librosa.load(str(input_data.audio), sr=22050, mono=True)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
            bpm = scalar_tempo(tempo)
            if bpm <= 0:
                bpm = DEFAULT_BPM
            return BpmOutput(bpm=bpm, beat_times_sec=beat_times, model_id=self.model_id)
        except Exception as exc:
            log.warning("BPM detection failed; using fallback %.0f: %s", DEFAULT_BPM, exc)
            return BpmOutput(bpm=DEFAULT_BPM, beat_times_sec=[], model_id=f"{self.model_id}+fallback")
