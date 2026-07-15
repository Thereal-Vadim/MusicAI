"""Model registry for inference adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from inference.adapters.base import BaseModelAdapter
from inference.adapters.basic_pitch_adapter import BasicPitchAdapter
from inference.adapters.bs_roformer_adapter import BSRoFormerAdapter
from inference.adapters.demucs_adapter import DemucsAdapter
from inference.adapters.librosa_bpm_adapter import LibrosaBpmAdapter
from inference.adapters.mediapipe_adapter import MediaPipeAdapter
from inference.adapters.wave_unet_adapter import WaveUNetAdapter
from inference.cloud.replicate_client import ReplicateDemucsAdapter
from inference.settings import InferenceSettings, inference_settings


class ModelRegistry:
    def __init__(self, settings: InferenceSettings | None = None) -> None:
        self.settings = settings or inference_settings
        self._adapters: dict[str, BaseModelAdapter] = {}

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        settings: InferenceSettings | None = None,
    ) -> "ModelRegistry":
        registry = cls(settings=settings)
        cfg = settings or inference_settings
        path = Path(config_path or cfg.models_config_path or Path(__file__).parent / "config" / "models.yaml")
        config = yaml.safe_load(path.read_text()) if path.exists() else {"models": {}}
        runtime = cfg.inference_runtime or config.get("runtime", "local")

        for model_id, spec in config.get("models", {}).items():
            adapter = registry._build_adapter(model_id, spec, runtime, cfg)
            if adapter is not None:
                registry._adapters[model_id] = adapter

        return registry

    @staticmethod
    def _build_adapter(
        model_id: str,
        spec: dict[str, Any],
        runtime: str,
        cfg: InferenceSettings,
    ) -> BaseModelAdapter | None:
        adapter_type = spec.get("adapter")

        if runtime == "cloud" and adapter_type == "demucs":
            return ReplicateDemucsAdapter(api_token=cfg.replicate_api_token)

        if adapter_type == "demucs":
            return DemucsAdapter(
                model_id=model_id,
                model_name=cfg.demucs_model or spec.get("model_name", "htdemucs_6s"),
                device=cfg.demucs_device or spec.get("device", "cpu"),
            )
        if adapter_type == "basic_pitch":
            return BasicPitchAdapter(
                model_id=model_id,
                onset_threshold=cfg.basic_pitch_onset_threshold,
                frame_threshold=cfg.basic_pitch_frame_threshold,
            )
        if adapter_type == "mediapipe":
            return MediaPipeAdapter(
                model_id=model_id,
                min_detection_confidence=cfg.mediapipe_min_detection_confidence,
                min_tracking_confidence=cfg.mediapipe_min_tracking_confidence,
            )
        if adapter_type == "librosa":
            return LibrosaBpmAdapter(model_id=model_id)
        if adapter_type == "bs_roformer":
            checkpoint = cfg.bs_roformer_checkpoint or spec.get("checkpoint")
            checkpoint_path = Path(checkpoint) if checkpoint else None
            return BSRoFormerAdapter(
                model_id=model_id,
                checkpoint_path=checkpoint_path,
                device=cfg.bs_roformer_device or spec.get("device", "cpu"),
                segment_size=cfg.bs_roformer_segment_size or spec.get("segment_size", 256),
            )
        if adapter_type == "wave_unet":
            weights = cfg.wave_unet_weights or spec.get("weights_path")
            weights_path = Path(weights) if weights else None
            return WaveUNetAdapter(
                model_id=model_id,
                weights_path=weights_path,
                device=cfg.wave_unet_device or spec.get("device", "cpu"),
            )
        return None

    def get(self, model_id: str) -> BaseModelAdapter:
        if model_id not in self._adapters:
            raise KeyError(f"Model not registered: {model_id}")
        return self._adapters[model_id]

    def list_models(self) -> list[str]:
        return list(self._adapters.keys())

    def healthcheck_all(self) -> dict[str, bool]:
        return {model_id: adapter.healthcheck() for model_id, adapter in self._adapters.items()}

    def describe_all(self) -> list[dict[str, object]]:
        return [adapter.describe() for adapter in self._adapters.values()]

    def runtime_config(self) -> dict[str, object]:
        return {
            "runtime": self.settings.inference_runtime,
            "demucs_model": self.settings.demucs_model,
            "demucs_device": self.settings.demucs_device,
            "basic_pitch_onset_threshold": self.settings.basic_pitch_onset_threshold,
            "basic_pitch_frame_threshold": self.settings.basic_pitch_frame_threshold,
            "mediapipe_min_detection_confidence": self.settings.mediapipe_min_detection_confidence,
            "mediapipe_min_tracking_confidence": self.settings.mediapipe_min_tracking_confidence,
            "cloud_configured": bool(self.settings.replicate_api_token),
            "bs_roformer_checkpoint": str(self.settings.bs_roformer_checkpoint)
            if self.settings.bs_roformer_checkpoint
            else None,
            "wave_unet_weights": str(self.settings.wave_unet_weights)
            if self.settings.wave_unet_weights
            else None,
            "fingering_optimizer": self.settings.fingering_optimizer,
        }
