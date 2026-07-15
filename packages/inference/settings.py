"""Inference runtime settings loaded from environment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    inference_runtime: Literal["local", "cloud"] = "local"
    demucs_model: str = "htdemucs_6s"
    demucs_device: str = "cpu"
    basic_pitch_onset_threshold: float = 0.5
    basic_pitch_frame_threshold: float = 0.3
    mediapipe_min_detection_confidence: float = 0.7
    mediapipe_min_tracking_confidence: float = 0.5
    replicate_api_token: str | None = None
    models_config_path: Path | None = None


inference_settings = InferenceSettings()
