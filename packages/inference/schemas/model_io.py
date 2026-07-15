"""Inference schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typing import Literal

from pydantic import BaseModel, Field

GuitarPart = Literal["combined", "solo", "rhythm"]
SeparateMode = Literal["two_stems", "multi_stem"]


class SeparateInput(BaseModel):
    audio: Path
    stem: str = "guitar"
    guitar_part: GuitarPart = "combined"
    mode: SeparateMode = "multi_stem"
    output_dir: Path | None = None


class SeparateOutput(BaseModel):
    stem_path: Path
    model_id: str
    guitar_part: GuitarPart = "combined"
    isolation_method: str | None = None
    coarse_stems: dict[str, Path] = Field(default_factory=dict)


class GuitarDemixInput(BaseModel):
    guitar_stem: Path
    output_dir: Path
    mix_path: Path | None = None


class GuitarDemixOutput(BaseModel):
    solo_path: Path
    rhythm_path: Path
    model_id: str
    method: str
    diagnostics: dict[str, float] = Field(default_factory=dict)


class TranscribeInput(BaseModel):
    audio: Path


class RawNoteEvent(BaseModel):
    pitch_midi: int
    start_sec: float
    duration_sec: float
    confidence: float = 0.5


class TranscribeOutput(BaseModel):
    notes: list[RawNoteEvent]
    model_id: str


class VisionInput(BaseModel):
    video: Path | None = None
    audio: Path | None = None
    sample_fps: float = 2.0


class HandLandmark(BaseModel):
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


class HandFrame(BaseModel):
    timestamp_ms: float
    landmarks: list[HandLandmark] = Field(default_factory=list)
    fret_zone: int | None = None
    visibility_score: float = 0.0
    fretboard_bbox: list[float] | None = None


class VisionOutput(BaseModel):
    frames: list[HandFrame]
    model_id: str
    fallback_audio_only: bool = False


class BpmInput(BaseModel):
    audio: Path


class BpmOutput(BaseModel):
    bpm: float
    beat_times_sec: list[float] = Field(default_factory=list)
    model_id: str


class ModelInput(BaseModel):
    payload: Any
