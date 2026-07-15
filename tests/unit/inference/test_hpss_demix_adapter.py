"""HPSS demix adapter tests."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from inference.adapters.hpss_demix_adapter import HpssDemixAdapter
from inference.schemas.model_io import GuitarDemixInput


@pytest.mark.asyncio
async def test_hpss_demix_writes_standard_stems(tmp_path: Path) -> None:
    sr = 44100
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    guitar = tmp_path / "guitar.wav"
    sf.write(str(guitar), tone, sr)

    adapter = HpssDemixAdapter()
    assert adapter.healthcheck() is True

    result = await adapter.predict(
        GuitarDemixInput(
            guitar_stem=guitar,
            output_dir=tmp_path / "demix",
        )
    )

    assert result.method == "hpss_stereo_v2"
    assert result.solo_path.name == "solo.wav"
    assert result.rhythm_path.name == "rhythm.wav"
    assert result.solo_path.stat().st_size > 0
    assert result.rhythm_path.stat().st_size > 0
