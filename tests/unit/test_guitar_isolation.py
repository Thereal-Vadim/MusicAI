from __future__ import annotations

import numpy as np
import soundfile as sf

from musicai_worker.guitar_isolation import (
    RHYTHM_CFG,
    _apply_side_rhythm_mask,
    demix_guitar_stems,
    isolate_guitar_part,
)


def test_isolate_combined_returns_source(tmp_path) -> None:
    src = tmp_path / "guitar.wav"
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100, endpoint=False))
    sf.write(str(src), tone.astype(np.float32), 44100)

    out = isolate_guitar_part(src, "combined", tmp_path / "parts")
    assert out == src


def test_isolate_solo_writes_new_file(tmp_path) -> None:
    src = tmp_path / "guitar.wav"
    sr = 44100
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    mix = 0.4 * np.sin(2 * np.pi * 330 * t) + 0.2 * np.sign(np.sin(2 * np.pi * 8 * t))
    sf.write(str(src), mix.astype(np.float32), sr)

    out = isolate_guitar_part(src, "solo", tmp_path / "parts")
    assert out != src
    assert out.name == "guitar_solo_v2.wav"
    assert out.exists()


def test_isolate_rhythm_writes_new_file(tmp_path) -> None:
    src = tmp_path / "guitar.wav"
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    mix = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sign(np.sin(2 * np.pi * 4 * t))
    sf.write(str(src), mix.astype(np.float32), sr)

    out = isolate_guitar_part(src, "rhythm", tmp_path / "parts")
    assert out.name == "guitar_rhythm_v2.wav"
    assert out.exists()


def test_side_mask_boosts_panned_content() -> None:
    sr = 44100
    n = sr
    t = np.linspace(0, 1.0, n, endpoint=False)
    panned = 0.8 * np.sin(2 * np.pi * 220 * t)
    centered = 0.8 * np.sin(2 * np.pi * 880 * t)
    stereo = np.stack([panned, -panned], axis=0)
    guitar = panned + centered

    masked = _apply_side_rhythm_mask(guitar.astype(np.float32), stereo, sr, RHYTHM_CFG)
    panned_energy = float(np.mean(masked**2))
    centered_only = _apply_side_rhythm_mask(centered.astype(np.float32), stereo, sr, RHYTHM_CFG)
    centered_energy = float(np.mean(centered_only**2))
    assert panned_energy > centered_energy * 1.5


def test_rhythm_emphasis_runs_on_stereo_mix(tmp_path) -> None:
    sr = 44100
    n = sr // 2
    t = np.linspace(0, 0.5, n, endpoint=False)
    stereo = np.stack([0.6 * np.sin(2 * np.pi * 220 * t), -0.6 * np.sin(2 * np.pi * 220 * t)])
    guitar = np.mean(stereo, axis=0)
    guitar_path = tmp_path / "guitar.wav"
    mix_path = tmp_path / "mix.wav"
    sf.write(str(guitar_path), guitar.astype(np.float32), sr)
    sf.write(str(mix_path), stereo.T, sr)

    out = isolate_guitar_part(
        guitar_path,
        "rhythm",
        tmp_path / "parts",
        mix_path=mix_path,
    )
    assert out.name == "guitar_rhythm_v2.wav"
    assert out.exists()


def test_demix_guitar_stems_writes_solo_and_rhythm(tmp_path) -> None:
    sr = 44100
    n = sr // 2
    t = np.linspace(0, 0.5, n, endpoint=False)
    stereo = np.stack([0.6 * np.sin(2 * np.pi * 220 * t), -0.6 * np.sin(2 * np.pi * 220 * t)])
    guitar = np.mean(stereo, axis=0)
    guitar_path = tmp_path / "guitar.wav"
    mix_path = tmp_path / "mix.wav"
    sf.write(str(guitar_path), guitar.astype(np.float32), sr)
    sf.write(str(mix_path), stereo.T, sr)

    solo, rhythm, diag = demix_guitar_stems(
        guitar_path,
        tmp_path / "demix",
        mix_path=mix_path,
    )
    assert solo.name == "solo.wav"
    assert rhythm.name == "rhythm.wav"
    assert solo.parent.name == "standard"
    assert solo.stat().st_size > 0
    assert rhythm.stat().st_size > 0
    assert diag["stereo_used"] == 1.0
