"""Spectral dereverberation adapter (guitar stem cleanup before transcription)."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from inference.adapters.base import BaseModelAdapter
from inference.audio.dereverb import dereverb_file
from inference.schemas.model_io import DereverbInput, DereverbOutput

log = logging.getLogger("musicai.spectral_dereverb")


class SpectralDereverbAdapter(BaseModelAdapter):
    """
    Late-reverb / delay tail suppression via STFT Wiener gain + transient remix.

    Demucs 4.x does not ship a Denoiser; this is the MVP stand-in that sharpens
    note attacks for Basic Pitch without new model weights.
    """

    def __init__(
        self,
        model_id: str = "spectral/dereverb",
        strength: float = 0.65,
        decay_ms: float = 80.0,
        floor: float = 0.12,
        transient_mix: float = 0.18,
    ) -> None:
        self.model_id = model_id
        self.runtime = "local"
        self.strength = strength
        self.decay_ms = decay_ms
        self.floor = floor
        self.transient_mix = transient_mix

    def healthcheck(self) -> bool:
        try:
            import librosa  # noqa: F401
            import soundfile  # noqa: F401

            return True
        except ImportError:
            return False

    async def predict(self, input_data: DereverbInput) -> DereverbOutput:
        return await asyncio.to_thread(self._run, input_data)

    def _run(self, input_data: DereverbInput) -> DereverbOutput:
        src = Path(input_data.audio)
        dst = Path(input_data.output_path)
        if not src.exists():
            raise FileNotFoundError(f"Dereverb input missing: {src}")

        try:
            diagnostics = dereverb_file(
                src,
                dst,
                strength=self.strength,
                decay_ms=self.decay_ms,
                floor=self.floor,
                transient_mix=self.transient_mix,
            )
            return DereverbOutput(
                audio_path=dst,
                model_id=self.model_id,
                method="spectral_late_reverb_wiener",
                diagnostics=diagnostics,
            )
        except Exception as exc:
            log.warning("Spectral dereverb failed (%s); copying original stem", exc)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            return DereverbOutput(
                audio_path=dst,
                model_id=f"{self.model_id}+passthrough",
                method="passthrough",
                diagnostics={"strength": 0.0, "energy_reduction": 0.0},
            )

    def describe(self) -> dict[str, object]:
        base = super().describe()
        base["backend"] = "spectral"
        base["strength"] = self.strength
        base["decay_ms"] = self.decay_ms
        base["transient_mix"] = self.transient_mix
        return base
