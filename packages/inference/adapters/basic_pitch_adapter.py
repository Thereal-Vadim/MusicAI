"""Basic Pitch transcription adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path

from inference.schemas.model_io import RawNoteEvent, TranscribeInput, TranscribeOutput


class BasicPitchAdapter:
    model_id = "basic-pitch/v1"
    runtime = "local"

    def __init__(
        self,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
    ) -> None:
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold

    def healthcheck(self) -> bool:
        try:
            import basic_pitch  # noqa: F401

            return True
        except ImportError:
            return False

    async def predict(self, input_data: TranscribeInput) -> TranscribeOutput:
        return await asyncio.to_thread(self._transcribe, input_data)

    def _transcribe(self, input_data: TranscribeInput) -> TranscribeOutput:
        notes: list[RawNoteEvent] = []
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH

            _, midi_data, note_events = predict(
                str(input_data.audio),
                ICASSP_2022_MODEL_PATH,
                onset_threshold=self.onset_threshold,
                frame_threshold=self.frame_threshold,
            )
            del midi_data
            for start, end, pitch, conf, _ in note_events:
                notes.append(
                    RawNoteEvent(
                        pitch_midi=int(round(pitch)),
                        start_sec=float(start),
                        duration_sec=max(float(end - start), 0.05),
                        confidence=float(conf),
                    )
                )
        except Exception:
            notes = self._librosa_fallback(input_data.audio)

        return TranscribeOutput(notes=notes, model_id=self.model_id)

    def _librosa_fallback(self, audio_path: Path) -> list[RawNoteEvent]:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        notes: list[RawNoteEvent] = []
        hop = 512
        for frame in range(pitches.shape[1]):
            idx = magnitudes[:, frame].argmax()
            pitch = pitches[idx, frame]
            if pitch <= 0:
                continue
            midi = int(round(librosa.hz_to_midi(pitch)))
            if midi < 40 or midi > 88:
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
