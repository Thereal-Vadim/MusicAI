"""Wave-U-Net guitar demix adapter (solo vs rhythm neural separation)."""

from __future__ import annotations

import asyncio
import gc
import logging
import shutil
import threading
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from inference.adapters.base import BaseModelAdapter
from inference.adapters.wave_unet_model import GuitarDemixWaveUNet
from inference.schemas.model_io import GuitarDemixInput, GuitarDemixOutput

log = logging.getLogger("musicai.wave_unet")

STANDARD_SOLO = "solo.wav"
STANDARD_RHYTHM = "rhythm.wav"
SAMPLE_RATE = 44100


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def release_accelerator_memory(device: str) -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if device == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


class _WaveUNetRuntime:
    """Lazy singleton — loads guitar demix weights on first inference only."""

    _instances: dict[tuple[str, str], "_WaveUNetRuntime"] = {}
    _lock = threading.Lock()

    def __init__(self, weights_path: Path, device: str) -> None:
        self.weights_path = weights_path
        self.device = device
        self._model: Any | None = None
        self._model_loaded = False

    @classmethod
    def get(cls, weights_path: Path, device: str) -> "_WaveUNetRuntime":
        key = (str(weights_path.resolve()), device)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(weights_path, device)
            return cls._instances[key]

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        import torch

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Wave-U-Net checkpoint not found at {self.weights_path}")

        log.info("Wave-U-Net lazy load weights=%s device=%s", self.weights_path.name, self.device)
        checkpoint = torch.load(str(self.weights_path), map_location=self.device, weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            model = GuitarDemixWaveUNet()
            if isinstance(checkpoint, dict):
                state = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
                if isinstance(state, dict):
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    if missing:
                        log.warning("Wave-U-Net missing keys=%d unexpected=%d", len(missing), len(unexpected))
                else:
                    raise RuntimeError("Wave-U-Net checkpoint dict has no state_dict")
            else:
                raise RuntimeError(f"Unsupported Wave-U-Net checkpoint type: {type(checkpoint)}")

        model.to(self.device)
        model.eval()
        self._model = model
        self._model_loaded = True
        return self._model

    @staticmethod
    def _overlap_add(
        chunks: list[np.ndarray],
        hop: int,
        total_length: int,
    ) -> np.ndarray:
        out = np.zeros(total_length, dtype=np.float32)
        weights = np.zeros(total_length, dtype=np.float32)
        window = np.hanning(hop * 2).astype(np.float32)
        left_win = window[:hop]
        right_win = window[hop:]

        for idx, chunk in enumerate(chunks):
            start = idx * hop
            end = start + len(chunk)
            if end > total_length:
                chunk = chunk[: total_length - start]
                end = total_length
            seg_len = end - start
            w = np.ones(seg_len, dtype=np.float32)
            if seg_len >= hop:
                w[:hop] *= left_win[: min(hop, seg_len)]
                if seg_len > hop:
                    w[-hop:] *= right_win[: min(hop, seg_len - hop)]
            out[start:end] += chunk[:seg_len] * w
            weights[start:end] += w

        weights = np.maximum(weights, 1e-8)
        return out / weights

    def separate(self, audio_path: Path, output_dir: Path) -> dict[str, Path]:
        import torch

        model = self._ensure_model()
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

        segment = SAMPLE_RATE * 12
        hop = segment // 2
        solo_chunks: list[np.ndarray] = []
        rhythm_chunks: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, max(1, len(y)), hop):
                chunk = y[start : start + segment]
                if chunk.size == 0:
                    continue
                if chunk.size < segment:
                    chunk = np.pad(chunk, (0, segment - chunk.size))

                tensor = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
                solo_t, rhythm_t = model(tensor)
                solo_np = solo_t.squeeze().detach().cpu().numpy()
                rhythm_np = rhythm_t.squeeze().detach().cpu().numpy()

                valid = min(len(chunk), len(solo_np), len(rhythm_np))
                solo_chunks.append(solo_np[:valid].astype(np.float32))
                rhythm_chunks.append(rhythm_np[:valid].astype(np.float32))

                if start + segment >= len(y):
                    break

        solo = self._overlap_add(solo_chunks, hop, len(y))
        rhythm = self._overlap_add(rhythm_chunks, hop, len(y))

        raw_dir = output_dir / "raw"
        standard_dir = output_dir / "standard"
        raw_dir.mkdir(parents=True, exist_ok=True)
        standard_dir.mkdir(parents=True, exist_ok=True)

        solo_raw = raw_dir / "extracted_solo.wav"
        rhythm_raw = raw_dir / "extracted_rhythm.wav"
        solo_std = standard_dir / STANDARD_SOLO
        rhythm_std = standard_dir / STANDARD_RHYTHM

        sf.write(str(solo_raw), solo, SAMPLE_RATE)
        sf.write(str(rhythm_raw), rhythm, SAMPLE_RATE)
        shutil.copy2(solo_raw, solo_std)
        shutil.copy2(rhythm_raw, rhythm_std)

        return {"solo": solo_std, "rhythm": rhythm_std}


class WaveUNetAdapter(BaseModelAdapter):
    """
    Neural guitar demix via Wave-U-Net.

    Lazy loads weights on first predict(). Requires WAVE_UNET_WEIGHTS — no heuristic fallback.
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
        self._runtime: _WaveUNetRuntime | None = None

    def healthcheck(self) -> bool:
        if not _torch_available():
            return False
        return self.weights_path is not None and self.weights_path.exists()

    async def predict(self, input_data: GuitarDemixInput) -> GuitarDemixOutput:
        return await asyncio.to_thread(self._demix, input_data)

    def _get_runtime(self) -> _WaveUNetRuntime:
        if self.weights_path is None:
            raise RuntimeError(f"{self.model_id}: weights path is not configured")
        if self._runtime is None:
            self._runtime = _WaveUNetRuntime.get(self.weights_path, self.device)
        return self._runtime

    def _demix(self, input_data: GuitarDemixInput) -> GuitarDemixOutput:
        if not self.healthcheck():
            raise RuntimeError(
                f"{self.model_id}: Wave-U-Net unavailable — set WAVE_UNET_WEIGHTS and install torch"
            )

        output_dir = input_data.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        runtime = self._get_runtime()

        try:
            stems = runtime.separate(input_data.guitar_stem, output_dir)
            solo_path = stems["solo"]
            rhythm_path = stems["rhythm"]
            solo, _ = librosa.load(str(solo_path), sr=SAMPLE_RATE, mono=True)
            rhythm, _ = librosa.load(str(rhythm_path), sr=SAMPLE_RATE, mono=True)
            diagnostics = {
                "solo_rms": float(np.sqrt(np.mean(solo**2))),
                "rhythm_rms": float(np.sqrt(np.mean(rhythm**2))),
                "model_loaded": float(runtime.model_loaded),
            }
        finally:
            release_accelerator_memory(self.device)

        log.info("Wave-U-Net demix complete diagnostics=%s", diagnostics)

        return GuitarDemixOutput(
            solo_path=solo_path,
            rhythm_path=rhythm_path,
            model_id=self.model_id,
            method="wave_unet_neural_v1",
            diagnostics=diagnostics,
        )

    def describe(self) -> dict[str, object]:
        base = super().describe()
        base["backend"] = "wave_unet"
        base["weights"] = str(self.weights_path) if self.weights_path else None
        base["device"] = self.device
        base["model_loaded"] = self._runtime.model_loaded if self._runtime else False
        return base
