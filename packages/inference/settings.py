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
    mediapipe_hand_model: Path | None = None
    replicate_api_token: str | None = None
    models_config_path: Path | None = None
    pipeline_config_path: Path | None = None

    # BS-RoFormer (coarse separation)
    bs_roformer_checkpoint: Path | None = None
    bs_roformer_device: str = "cpu"
    bs_roformer_segment_size: int = 256

    # Wave-U-Net (guitar demix)
    wave_unet_weights: Path | None = None
    wave_unet_device: str = "cpu"

    # Spectral dereverb (guitar stem cleanup)
    dereverb_enabled: bool = True
    dereverb_strength: float = 0.65
    dereverb_decay_ms: float = 80.0
    dereverb_floor: float = 0.12
    dereverb_transient_mix: float = 0.18

    # Fingering optimizer
    fingering_optimizer: Literal["aco", "dp"] = "dp"
    aco_n_ants: int = 24
    aco_n_iterations: int = 50
    aco_max_fret_span: int = 4


inference_settings = InferenceSettings()
