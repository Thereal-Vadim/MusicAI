"""Basic Pitch transcription adapter."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from inference.adapters.base import BaseModelAdapter
from inference.schemas.model_io import RawNoteEvent, TranscribeInput, TranscribeOutput

log = logging.getLogger("musicai.basic_pitch")


class BasicPitchAdapter(BaseModelAdapter):
    def __init__(
        self,
        model_id: str = "basic-pitch/v1",
        onset_threshold: float = 0.65,
        frame_threshold: float = 0.45,
        minimum_note_length_ms: float = 50.0,
        minimum_frequency: float = 70.0,
        maximum_frequency: float = 1200.0,
    ) -> None:
        self.model_id = model_id
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length_ms = minimum_note_length_ms
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.runtime = "local"

    def healthcheck(self) -> bool:
        try:
            import basic_pitch  # noqa: F401

            return True
        except Exception:
            return False

    async def predict(self, input_data: TranscribeInput) -> TranscribeOutput:
        return await asyncio.to_thread(self._transcribe, input_data)

    def _transcribe(self, input_data: TranscribeInput) -> TranscribeOutput:
        notes: list[RawNoteEvent] = []
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH

            note_events = self._run_predict(str(input_data.audio), ICASSP_2022_MODEL_PATH)
            for start, end, pitch, conf, _ in note_events:
                notes.append(
                    RawNoteEvent(
                        pitch_midi=int(round(pitch)),
                        start_sec=float(start),
                        duration_sec=max(float(end - start), 0.05),
                        confidence=float(conf),
                    )
                )
        except Exception as exc:
            log.warning("Basic Pitch failed (%s), using librosa fallback", exc)
            notes = self._librosa_fallback(input_data.audio)

        return TranscribeOutput(notes=notes, model_id=self.model_id)

    def _run_predict(self, audio_path: str, model_path: object) -> list:
        from basic_pitch.inference import predict

        kwargs = {
            "onset_threshold": self.onset_threshold,
            "frame_threshold": self.frame_threshold,
            "minimum_note_length": self.minimum_note_length_ms,
            "minimum_frequency": self.minimum_frequency,
            "maximum_frequency": self.maximum_frequency,
        }
        try:
            _, midi_data, note_events = predict(audio_path, model_path, **kwargs)
        except TypeError as exc:
            log.warning(
                "Basic Pitch reject extended kwargs (%s); using onset/frame only",
                exc,
            )
            _, midi_data, note_events = predict(
                audio_path,
                model_path,
                onset_threshold=self.onset_threshold,
                frame_threshold=self.frame_threshold,
            )
        del midi_data
        return note_events

    def _librosa_fallback(self, audio_path: Path) -> list[RawNoteEvent]:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=sr,
            fmin=self.minimum_frequency,
            fmax=self.maximum_frequency,
        )
        notes: list[RawNoteEvent] = []
        hop = 512
        min_midi = int(round(librosa.hz_to_midi(self.minimum_frequency)))
        max_midi = int(round(librosa.hz_to_midi(self.maximum_frequency)))
        for frame in range(pitches.shape[1]):
            idx = magnitudes[:, frame].argmax()
            pitch = pitches[idx, frame]
            if pitch <= 0:
                continue
            midi = int(round(librosa.hz_to_midi(pitch)))
            if midi < min_midi or midi > max_midi:
                continue
            start_sec = frame * hop / sr
            notes.append(
                RawNoteEvent(
                    pitch_midi=midi,
                    start_sec=start_sec,
                    duration_sec=hop / sr,
                    confidence=0.35,
                )
            )
        if not notes:
            notes.append(RawNoteEvent(pitch_midi=64, start_sec=0.0, duration_sec=0.25, confidence=0.2))
        return self._dedupe_notes(notes)

    @staticmethod
    def _dedupe_notes(notes: list[RawNoteEvent]) -> list[RawNoteEvent]:
        if not notes:
            return notes
        merged: list[RawNoteEvent] = [notes[0]]
        for note in notes[1:]:
            prev = merged[-1]
            if abs(note.start_sec - prev.start_sec) < 0.08 and note.pitch_midi == prev.pitch_midi:
                prev.duration_sec = max(prev.duration_sec, note.duration_sec)
                prev.confidence = max(prev.confidence, note.confidence)
            else:
                merged.append(note)
        return merged

    def describe(self) -> dict[str, object]:
        base = super().describe()
        base["onset_threshold"] = self.onset_threshold
        base["frame_threshold"] = self.frame_threshold
        base["minimum_note_length_ms"] = self.minimum_note_length_ms
        base["minimum_frequency"] = self.minimum_frequency
        base["maximum_frequency"] = self.maximum_frequency
        return base
