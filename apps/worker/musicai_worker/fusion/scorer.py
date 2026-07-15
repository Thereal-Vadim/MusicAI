"""Fusion of audio, vision, theory signals, and tab benchmark scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inference.schemas.model_io import HandFrame, RawNoteEvent
from tab_schema.alignment import (
    TabAlignmentResult,
    align_documents,
    align_files,
    align_tab_notes,
    load_notes_from_document,
    load_notes_from_path,
)
from tab_schema.models import TabDocument, TabNote


@dataclass
class FusionConfig:
    vision_weight: float = 0.25
    audio_weight: float = 0.75
    vision_mismatch_threshold: int = 2


class FusionScorer:
    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def fuse_notes(
        self,
        notes: list[TabNote],
        vision_frames: list[HandFrame],
        audio_only: bool,
    ) -> list[TabNote]:
        if audio_only or not vision_frames:
            for note in notes:
                note.confidence.overall = note.confidence.audio
            return notes

        for note in notes:
            frame = self._nearest_frame(note.start_ms, vision_frames)
            if frame is None or frame.fret_zone is None:
                note.confidence.overall = note.confidence.audio
                continue

            note.confidence.vision = frame.visibility_score
            vision_fret = frame.fret_zone
            note.sources.vision_fret = vision_fret
            note.sources.audio_fret = note.fret

            if abs(vision_fret - note.fret) > self.config.vision_mismatch_threshold:
                note.flags.append("audio_vision_mismatch")
                if frame.visibility_score > 0.6:
                    note.fret = round(
                        note.fret * self.config.audio_weight
                        + vision_fret * self.config.vision_weight
                    )
            note.confidence.overall = (
                note.confidence.audio * self.config.audio_weight
                + note.confidence.vision * self.config.vision_weight
            )

        return notes

    @staticmethod
    def _nearest_frame(timestamp_ms: float, frames: list[HandFrame]) -> HandFrame | None:
        if not frames:
            return None
        return min(frames, key=lambda f: abs(f.timestamp_ms - timestamp_ms))

    @staticmethod
    def raw_to_tab_notes(raw_notes: list[RawNoteEvent]) -> list[tuple[int, float, float, float]]:
        return [
            (
                n.pitch_midi,
                n.start_sec * 1000.0,
                n.duration_sec * 1000.0,
                n.confidence,
            )
            for n in raw_notes
        ]

    def benchmark_against_reference(
        self,
        predicted: TabDocument,
        reference: TabDocument | Path,
        *,
        window_ms: float = 180.0,
        timing_tolerance_ms: float = 80.0,
    ) -> TabAlignmentResult:
        """Align predicted tab vs ground truth; returns pitch/fret/string/timing metrics."""
        if isinstance(reference, Path):
            return align_tab_notes(
                load_notes_from_path(reference),
                load_notes_from_document(predicted),
                window_ms=window_ms,
                timing_tolerance_ms=timing_tolerance_ms,
            )
        return align_documents(
            reference,
            predicted,
            window_ms=window_ms,
            timing_tolerance_ms=timing_tolerance_ms,
        )

    def benchmark_files(
        self,
        reference_path: Path,
        predicted_path: Path,
        *,
        window_ms: float = 180.0,
        timing_tolerance_ms: float = 80.0,
    ) -> TabAlignmentResult:
        return align_files(
            reference_path,
            predicted_path,
            window_ms=window_ms,
            timing_tolerance_ms=timing_tolerance_ms,
        )
