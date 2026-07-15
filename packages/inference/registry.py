"""Model registry for inference adapters."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from inference.adapters.basic_pitch_adapter import BasicPitchAdapter
from inference.adapters.demucs_adapter import DemucsAdapter
from inference.adapters.librosa_bpm_adapter import LibrosaBpmAdapter
from inference.adapters.mediapipe_adapter import MediaPipeAdapter
from inference.cloud.replicate_client import ReplicateDemucsAdapter


class ModelRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, object] = {}

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "ModelRegistry":
        registry = cls()
        path = Path(config_path or Path(__file__).parent / "config" / "models.yaml")
        config = yaml.safe_load(path.read_text()) if path.exists() else {"models": {}}
        runtime = os.getenv("INFERENCE_RUNTIME", config.get("runtime", "local"))

        for model_id, spec in config.get("models", {}).items():
            adapter_type = spec.get("adapter")
            if runtime == "cloud" and adapter_type == "demucs":
                registry._adapters[model_id] = ReplicateDemucsAdapter(
                    api_token=os.getenv("REPLICATE_API_TOKEN")
                )
                continue

            if adapter_type == "demucs":
                registry._adapters[model_id] = DemucsAdapter(
                    model_name=os.getenv("DEMUCS_MODEL", spec.get("model_name", "htdemucs_6s")),
                    device=os.getenv("DEMUCS_DEVICE", spec.get("device", "cpu")),
                )
            elif adapter_type == "basic_pitch":
                registry._adapters[model_id] = BasicPitchAdapter(
                    onset_threshold=float(
                        os.getenv("BASIC_PITCH_ONSET_THRESHOLD", spec.get("onset_threshold", 0.5))
                    ),
                    frame_threshold=float(
                        os.getenv("BASIC_PITCH_FRAME_THRESHOLD", spec.get("frame_threshold", 0.3))
                    ),
                )
            elif adapter_type == "mediapipe":
                registry._adapters[model_id] = MediaPipeAdapter(
                    min_detection_confidence=float(
                        os.getenv(
                            "MEDIAPIPE_MIN_DETECTION_CONFIDENCE",
                            spec.get("min_detection_confidence", 0.7),
                        )
                    ),
                    min_tracking_confidence=float(
                        os.getenv(
                            "MEDIAPIPE_MIN_TRACKING_CONFIDENCE",
                            spec.get("min_tracking_confidence", 0.5),
                        )
                    ),
                )
            elif adapter_type == "librosa":
                registry._adapters[model_id] = LibrosaBpmAdapter()

        return registry

    def get(self, model_id: str) -> object:
        if model_id not in self._adapters:
            raise KeyError(f"Model not registered: {model_id}")
        return self._adapters[model_id]

    def list_models(self) -> list[str]:
        return list(self._adapters.keys())

    def healthcheck_all(self) -> dict[str, bool]:
        return {
            model_id: getattr(adapter, "healthcheck")()
            for model_id, adapter in self._adapters.items()
        }
