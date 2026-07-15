"""CASA guitar demix v2 tests."""

from pathlib import Path

import numpy as np
import soundfile as sf

from musicai_worker.guitar_demix import CASADemixConfig, demix_guitar_stem


def test_demix_v2_writes_solo_and_rhythm(tmp_path: Path) -> None:
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    guitar = tmp_path / "guitar.wav"
    sf.write(str(guitar), tone, sr)

    stereo_mix = tmp_path / "mix.wav"
    sf.write(str(stereo_mix), np.vstack([tone, tone * 0.5]).T, sr)

    result = demix_guitar_stem(
        guitar,
        tmp_path / "demix",
        mix_path=stereo_mix,
        config=CASADemixConfig(hpss_n_fft=1024, hpss_win_length=1024, n_fft=1024, win_length=1024),
    )

    assert result.solo.exists()
    assert result.rhythm.exists()
    assert result.method == "casa_wiener_hpss_spatial_v2"
    assert result.solo.stat().st_size > 0
    assert "pitch_confidence_mean" in result.diagnostics


def test_hard_gate_zeros_weak_mask() -> None:
    from musicai_worker.guitar_demix import _hard_gate

    mask = np.array([0.1, 0.6, 0.9], dtype=np.float32)
    gated = _hard_gate(mask, 0.55)
    assert gated[0] == 0.0
    assert gated[1] == 0.6
    assert gated[2] == 0.9
