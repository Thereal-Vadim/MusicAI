"""Judge module settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class JudgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    judge_snap_audio_confidence_threshold: float = 0.5
    judge_high_confidence_threshold: float = 0.95
    judge_snap_confidence_bonus: float = 0.08
    judge_beats_per_measure: int = 4
    judge_min_note_duration_ms: float = 50.0
    judge_max_simultaneous_notes: int = 4
    judge_max_chord_span: int = 5
    judge_use_music21: bool = True


judge_settings = JudgeSettings()
