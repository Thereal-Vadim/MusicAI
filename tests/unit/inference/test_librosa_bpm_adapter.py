"""Librosa BPM adapter tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from inference.adapters.librosa_bpm_adapter import DEFAULT_BPM, LibrosaBpmAdapter, scalar_tempo
from inference.schemas.model_io import BpmInput


def test_scalar_tempo_from_ndarray():
    assert scalar_tempo(np.array([123.046875])) == pytest.approx(123.046875)


def test_scalar_tempo_from_scalar():
    assert scalar_tempo(98.5) == pytest.approx(98.5)


def test_scalar_tempo_empty_defaults():
    assert scalar_tempo(np.array([])) == DEFAULT_BPM


def _write_click_track(path: Path, *, bpm: float, duration_sec: float = 8.0, sr: int = 22050) -> None:
    beat_interval = 60.0 / bpm
    t = np.arange(0, duration_sec, 1 / sr)
    y = np.zeros_like(t)
    for beat in np.arange(0, duration_sec, beat_interval):
        idx = int(beat * sr)
        if idx + 100 < len(y):
            y[idx : idx + 100] = 0.5 * np.sin(2 * np.pi * 440 * np.arange(100) / sr)
    sf.write(path, y, sr)


def test_detect_bpm_from_synthetic_click_track():
    adapter = LibrosaBpmAdapter()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "click.wav"
        _write_click_track(audio, bpm=123.0)
        result = adapter._detect(BpmInput(audio=audio))

    assert result.model_id == "librosa/beat"
    assert result.bpm == pytest.approx(123.0, rel=0.05)
    assert len(result.beat_times_sec) > 0


def test_detect_missing_file_falls_back():
    adapter = LibrosaBpmAdapter()
    result = adapter._detect(BpmInput(audio=Path("/tmp/does-not-exist-musicai-bpm.wav")))

    assert result.bpm == DEFAULT_BPM
    assert result.model_id == "librosa/beat+fallback"
    assert result.beat_times_sec == []
