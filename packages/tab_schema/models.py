"""Shared TabDocument Pydantic models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceMeta(BaseModel):
    type: Literal["upload", "youtube"]
    url: str | None = None
    youtube_id: str | None = None
    filename: str | None = None


class TabMeta(BaseModel):
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    bpm: float = 120.0
    key: str | None = None
    mode: str | None = None
    tuning: list[str] = Field(default_factory=lambda: ["E2", "A2", "D3", "G3", "B3", "E4"])
    guitar_part: Literal["combined", "solo", "rhythm"] = "combined"
    source: SourceMeta
    pipeline_version: str = "0.1.0"
    overall_confidence: float = 0.0
    quality: "QualityMeta | None" = None


class QualityMeta(BaseModel):
    """Actionable transcription quality — penalties only vs Songsterr reference when available."""

    notes_total: int = 0
    snapped_count: int = 0
    high_confidence_count: int = 0
    conflict_count: int = 0
    snapped_pct: float = 0.0
    high_confidence_pct: float = 0.0
    conflict_pct: float = 0.0
    mean_overall: float = 0.0
    key_confidence: float = 0.0
    reference_url: str | None = None
    reference_match_pct: float | None = None
    reference_mismatch_count: int = 0


class NoteTechnique(BaseModel):
    palm_mute: bool = False
    slide: Literal["up", "down", "into_from_below", "into_from_above"] | None = None
    vibrato: bool = False
    tie: bool = False
    ghost: bool = False


class NoteConfidence(BaseModel):
    audio: float = 0.0
    vision: float = 0.0
    judge: float = 1.0
    overall: float = 0.0


class NoteSources(BaseModel):
    audio_fret: int | None = None
    vision_fret: int | None = None
    theory_fret: int | None = None


class JudgeResult(BaseModel):
    in_scale: bool = True
    in_chord: bool = True
    snapped: bool = False
    snap_reason: str | None = None
    flags: list[str] = Field(default_factory=list)


class TabNote(BaseModel):
    id: str
    pitch: str
    original_pitch: str | None = None
    start_ms: float
    duration_ms: float
    string: int = Field(ge=1, le=6)
    fret: int = Field(ge=0, le=24)
    pitch_midi: int | None = None
    confidence: NoteConfidence = Field(default_factory=NoteConfidence)
    sources: NoteSources = Field(default_factory=NoteSources)
    judge: JudgeResult = Field(default_factory=JudgeResult)
    flags: list[str] = Field(default_factory=list)
    technique: NoteTechnique | None = None


class TabMeasure(BaseModel):
    index: int = 0
    start_ms: float
    confidence: float = 1.0
    chord: str | None = None
    time_signature: tuple[int, int] | None = None
    section: str | None = None
    tempo_bpm: float | None = None
    notes: list[TabNote] = Field(default_factory=list)


class TabTrack(BaseModel):
    instrument: Literal["guitar"] = "guitar"
    name: str | None = None
    midi_program: int | None = None
    role: Literal["solo", "rhythm", "combined"] | None = None
    measures: list[TabMeasure] = Field(default_factory=list)


class EditRecord(BaseModel):
    timestamp: str
    note_id: str
    field: str
    old_value: object
    new_value: object


class TabDocument(BaseModel):
    version: Literal["1"] = "1"
    job_id: str | None = None
    meta: TabMeta
    tracks: list[TabTrack] = Field(default_factory=list)
    edit_history: list[EditRecord] = Field(default_factory=list)

    def all_notes(self) -> list[TabNote]:
        notes: list[TabNote] = []
        for track in self.tracks:
            for measure in track.measures:
                notes.extend(measure.notes)
        return notes

    def conflict_note_ids(self) -> list[str]:
        ids: list[str] = []
        for note in self.all_notes():
            if note.flags or note.judge.flags or note.judge.snapped:
                if note.confidence.overall < 0.75 or note.judge.snapped:
                    ids.append(note.id)
            if "audio_vision_mismatch" in note.flags:
                ids.append(note.id)
        return list(dict.fromkeys(ids))
