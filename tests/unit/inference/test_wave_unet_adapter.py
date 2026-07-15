"""Wave-U-Net adapter tests."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from inference.adapters.wave_unet_adapter import (
    WaveUNetAdapter,
    _WaveUNetRuntime,
    release_accelerator_memory,
)
from inference.adapters.wave_unet_model import GuitarDemixWaveUNet
from inference.schemas.model_io import GuitarDemixInput


def test_wave_unet_unhealthy_without_weights() -> None:
    adapter = WaveUNetAdapter(weights_path=Path("/nonexistent/weights.pt"))
    assert adapter.healthcheck() is False


def test_runtime_singleton() -> None:
    path = Path("/tmp/wave_unet_test.pt")
    a = _WaveUNetRuntime.get(path, "cpu")
    b = _WaveUNetRuntime.get(path, "cpu")
    assert a is b
    assert not a.model_loaded


@pytest.mark.asyncio
async def test_wave_unet_inference_writes_standard_stems(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    weights = tmp_path / "guitar_demix.pt"
    model = GuitarDemixWaveUNet()
    torch.save(model.state_dict(), weights)

    sr = 44100
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    guitar = tmp_path / "guitar.wav"
    sf.write(str(guitar), tone, sr)

    adapter = WaveUNetAdapter(weights_path=weights, device="cpu")
    assert adapter.healthcheck() is True

    result = await adapter.predict(
        GuitarDemixInput(
            guitar_stem=guitar,
            output_dir=tmp_path / "demix",
        )
    )

    assert result.method == "wave_unet_neural_v1"
    assert result.solo_path.name == "solo.wav"
    assert result.rhythm_path.name == "rhythm.wav"
    assert result.solo_path.parent.name == "standard"
    assert result.solo_path.stat().st_size > 0
    assert result.rhythm_path.stat().st_size > 0
    release_accelerator_memory("cpu")
