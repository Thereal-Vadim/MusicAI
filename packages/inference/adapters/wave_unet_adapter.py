"""Wave-U-Net guitar demix adapter (solo vs rhythm polyphonic split)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from inference.adapters.base import BaseModelAdapter
from inference.schemas.model_io import GuitarDemixInput, GuitarDemixOutput

log = logging.getLogger("musicai.wave_unet")


class WaveUNetAdapter(BaseModelAdapter):
    """
    Wave-U-Net for polyphonic guitar voice separation.

    When `weights_path` is missing, healthcheck() returns False and pipeline routing
    falls back to the CASA heuristic demix in the worker.
    """

    def __init__(
        self,
        model_id: str = "wave-unet/guitar-demix",
        weights_path: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.weights_path = weights_path
        self.device = device
        self.runtime = "local"

    def healthcheck(self) -> bool:
        return self.weights_path is not None and self.weights_path.exists()

    async def predict(self, input_data: GuitarDemixInput) -> GuitarDemixOutput:
        return await asyncio.to_thread(self._demix, input_data)

    def _demix(self, input_data: GuitarDemixInput) -> GuitarDemixOutput:
        if not self.healthcheck():
            raise RuntimeError(
                f"{self.model_id}: Wave-U-Net weights not found at {self.weights_path}"
            )

        output_dir = input_data.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        solo_path = output_dir / "solo_wave_unet_v1.wav"
        rhythm_path = output_dir / "rhythm_wave_unet_v1.wav"

        y, sr = librosa.load(str(input_data.guitar_stem), sr=44100, mono=True)
        solo, rhythm = self._run_model(y, sr)

        sf.write(str(solo_path), solo, sr)
        sf.write(str(rhythm_path), rhythm, sr)

        diagnostics = {
            "solo_rms": float(np.sqrt(np.mean(solo**2))),
            "rhythm_rms": float(np.sqrt(np.mean(rhythm**2))),
            "input_rms": float(np.sqrt(np.mean(y**2))),
        }
        log.info("Wave-U-Net demix complete diagnostics=%s", diagnostics)

        return GuitarDemixOutput(
            solo_path=solo_path,
            rhythm_path=rhythm_path,
            model_id=self.model_id,
            method="wave_unet_polyphonic",
            diagnostics=diagnostics,
        )

    def _run_model(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Neural inference hook — load custom Wave-U-Net checkpoint here.

        Expects a model that outputs two mono stems (solo, rhythm) from guitar input.
        """
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Wave-U-Net requires torch — install with pip install torch") from exc

        raise NotImplementedError(
            "Wave-U-Net weights are configured but inference is not yet wired. "
            "Implement _run_model() with your trained checkpoint, or use CASA fallback."
        )

    def describe(self) -> dict[str, object]:
        base = super().describe()
        base["backend"] = "wave_unet"
        base["weights"] = str(self.weights_path) if self.weights_path else None
        return base
