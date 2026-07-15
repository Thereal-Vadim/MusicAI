"""Librosa BPM adapter."""

from __future__ import annotations

import asyncio

from inference.schemas.model_io import BpmInput, BpmOutput


class LibrosaBpmAdapter:
    model_id = "librosa/beat"
    runtime = "local"

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
            bpm = float(tempo) if hasattr(tempo, "__float__") else float(tempo[0])
            if bpm <= 0:
                bpm = 120.0
            return BpmOutput(bpm=bpm, beat_times_sec=beat_times, model_id=self.model_id)
        except Exception:
            return BpmOutput(bpm=120.0, beat_times_sec=[], model_id=f"{self.model_id}+fallback")
