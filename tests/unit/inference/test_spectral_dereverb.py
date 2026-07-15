"""Spectral dereverb + noise gate unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from inference.adapters.spectral_dereverb_adapter import SpectralDereverbAdapter
from inference.audio.dereverb import apply_noise_gate, spectral_dereverb
from inference.pipeline_routing import load_pipeline_routing
from inference.registry import ModelRegistry
from inference.schemas.model_io import DereverbInput


def _synthetic_reverb(y: np.ndarray, sr: int, decay_ms: float = 120.0) -> np.ndarray:
    """Simple exponential FIR tail to simulate room reverb."""
    delay = int(0.04 * sr)
    n = int(decay_ms / 1000.0 * sr)
    ir = np.zeros(delay + n, dtype=np.float32)
    ir[0] = 1.0
    t = np.arange(n) / sr
    ir[delay:] = 0.35 * np.exp(-t / (decay_ms / 1000.0))
    wet = np.convolve(y, ir, mode="full")[: len(y)]
    return (0.7 * y + 0.3 * wet).astype(np.float32)


def test_spectral_dereverb_reduces_tail_energy():
    sr = 22050
    t = np.arange(0, 1.5, 1 / sr)
    dry = (0.5 * np.sin(2 * np.pi * 220 * t) * np.exp(-t * 8)).astype(np.float32)
    wet = _synthetic_reverb(dry, sr)
    cleaned = spectral_dereverb(wet, sr, strength=0.8, transient_mix=0.1)

    late = slice(int(1.2 * sr), None)
    assert float(np.linalg.norm(cleaned[late])) < float(np.linalg.norm(wet[late]))


def test_spectral_dereverb_passthrough_zero_strength():
    sr = 22050
    y = np.random.randn(sr).astype(np.float32) * 0.1
    out = spectral_dereverb(y, sr, strength=0.0)
    assert np.allclose(out, y)


def test_noise_gate_zeros_quiet_floor_keeps_peaks():
    audio = np.array([0.0, 0.01, 0.02, 1.0, -0.9, 0.03], dtype=np.float32)
    gated = apply_noise_gate(audio, threshold=0.08)
    assert gated[0] == 0.0
    assert gated[1] == 0.0
    assert gated[2] == 0.0
    assert gated[3] == pytest.approx(1.0)
    assert gated[4] == pytest.approx(-0.9)
    assert gated[5] == 0.0


@pytest.mark.asyncio
async def test_adapter_writes_wav(tmp_path: Path):
    sr = 22050
    y = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    src = tmp_path / "in.wav"
    dst = tmp_path / "out.wav"
    sf.write(src, y, sr)

    adapter = SpectralDereverbAdapter(strength=0.5, gate_threshold=0.08)
    assert adapter.healthcheck() is True
    result = await adapter.predict(DereverbInput(audio=src, output_path=dst))

    assert result.audio_path.exists()
    assert result.method == "spectral_wiener+noise_gate"
    assert result.diagnostics["strength"] == pytest.approx(0.5)
    assert result.diagnostics["gate_threshold"] == pytest.approx(0.08)
    assert "gated_sample_fraction" in result.diagnostics


def test_registry_and_routing_include_audio_cleanup():
    routing = load_pipeline_routing()
    assert routing.audio_cleanup.enabled is True
    assert routing.audio_cleanup.primary == "spectral/dereverb"

    registry = ModelRegistry.from_config()
    assert "spectral/dereverb" in registry.list_models()
    assert registry.get("spectral/dereverb").healthcheck() is True
