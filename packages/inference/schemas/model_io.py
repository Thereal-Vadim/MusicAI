"""Inference schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SeparateInput(BaseModel):
    audio: Path
    stem: str = "guitar"
    output_dir: Path | None = None


class SeparateOutput(BaseModel):
    stem_path: Path
    model_id: str


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
